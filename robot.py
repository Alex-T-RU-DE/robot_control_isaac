from typing import Union, Tuple
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.core.robots import Robot
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.types import ArticulationAction
from .ik_solver import KinematicsSolver
import numpy as np
from scipy.spatial.transform import Rotation as R


class Robot:
    def __init__(self,
                 world_object: World = None,
                 end_effector_link_name: str = None,
                 urdf_path: str = None,
                 usd_path: str = None,
                 robot_descriptor_path: str = None,
                 rmpflow_config_path: str = None) -> None:
        """Creates robot object, which allows to use typical robot functions like moveJ,
        get telemetry and ik/fk calculation.

        Args:
            world_object (World, optional): Existing World object. If it does not yet exists,
                the new one will be created. Defaults to None.
            end_effector_link_name (str, optional): Link name of the TCP. should be defined in URDF.
            urdf_path (str, optional): path to robot's urdf file.
            usd_path (str, optional): path to robot's usd file.
            robot_descriptor_path (str, optional): path to robot descriptor yaml file.
            rmpflow_config_path (str, optional): path to rmpflow congig yaml.
        """
        self.urdf_path = urdf_path
        self.usd_path = usd_path
        self.robot_description_path = robot_descriptor_path
        self.rmp_config_path = rmpflow_config_path
        self.end_effector_name = end_effector_link_name
        self.robot_object = None
        self.rmpflow_object = None
        self.ik_object = None
        self.kinematics_solver = None
        self.world_object = world_object
        self.articulation_controller = None
        self.articulation_rmpflow = None
        self.articulation_subset = None
        self._spawn_robot()
        self.world_object.reset()
        return

    def _spawn_robot(self, prim_path: str, name: str = "robot") -> None:
        """Swapns the robot in the world and starts all important robotics
        controllers to be able to recieve robot's telemetry and to be able to
        send commands to the robot.

        Creates: KinematicsSolver objects, rmpflow objects, articulation controllers
        and World in case it does not yet exists.

        Args:
            prim_path (str): prim path, where the robot will be added in simulation.
            name (str, optional): shortname to be used as a key by Scene class. 
                Note: needs to be unique if the object is added to the Scene. Defaults to "robot".
        """
        open_stage(usd_path=self.usd_path)
        if self.world_object is None:
            self.world_object = World(stage_units_in_meters=1.0)
        self.robot_object = self.world_object.scene.add(Robot(prim_path=prim_path,
                                                              name=name))
        self.world_object.reset()
        self.rmpflow_object = RmpFlow(robot_description_path=self.robot_description_path,
                                      rmpflow_config_path=self.rmp_config_path,
                                      urdf_path=self.urdf_path,
                                      end_effector_frame_name=self.end_effector_name,
                                      evaluations_per_frame=5)
        self.articulation_rmpflow = ArticulationMotionPolicy(self.robot_object,
                                                             self.rmpflow_object, 1/60)
        self.articulation_controller = self.robot_object.get_articulation_controller()
        self.ik_object = KinematicsSolver(self.robot_object,
                                          urdf_path=self.urdf_path,
                                          robot_description_path=self.robot_description_path,
                                          end_effector_frame_name=self.end_effector_name)
        self.articulation_subset = self.ik_object.get_joints_subset()
        self.kinematics_solver = self.ik_object.get_kinematics_solver()

    def _get_pose_threshold(self, target: Union[list, np.array]) -> Tuple[float, float]:
        """Calculates differences between norms of given and current positions and orientations. 

        Args:
            target (Union[list, np.array]): target position and orientation

        Returns:
            Tuple[float, float]: difference between norms of given and current positions and orientations.
        """
        current_pose = self.get_tcp_pose()
        position_thresh = abs(np.linalg.norm(
            target[0]) - np.linalg.norm(current_pose[0]))
        orietntation = R.from_matrix(current_pose[1])
        orientation_thresh = abs(np.linalg.norm(
            target[1]) - np.linalg.norm(R.as_quat(orietntation)))
        return position_thresh, orientation_thresh

    def move_to_target(self, target: Union[list, np.array], timeout: int = 0) -> bool:
        """Moves robot to a given target. This motion relies on rmpflow motion generator.
        Similar to moveJ command.

        Note: if your robot can't achieve given position, or behaves unpredictlebly - 
        consider parameters' tuning in rmpflow config as mentioned in the link below

        https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_motion_generation.html#isaac-sim-motion-generation-rmpflow-tuning-guide

        Args:
            target (Union[list, np.array]): list/np.array of position (np.array(3x1))
                and orientation (quaternion np.array(4x1))
            timeout (int, optional): motion execution timeout in seconds.
                If set to 0 - no timeout. Default to 0.
        """
        # add exec time restriction
        position_thresh, orientation_thresh = self._get_pose_threshold(target)
        while (position_thresh > 0.001 or orientation_thresh > 0.001):
            self.world.step(render=True)
            if self.world.current_time_step_index == 0:
                self.world.reset()
            self.rmpflow_object.set_end_effector_target(
                target_position=target[0],
                target_orientation=target[1]
            )
            actions = self.articulation_rmpflow.get_next_articulation_action()
            self.articulation_controller.apply_action(actions)
            position_thresh, orientation_thresh = self._get_pose_threshold(
                target)
            #print(position_thresh, orientation_thresh)

    def get_joint_positions(self) -> list:
        """Returns angles of each joint in robot_desctiption.yaml

        Returns:
            list: joint positions in radians
        """
        return self.articulation_subset.get_joint_positions()

    def get_tcp_pose(self) -> Tuple[np.array, np.array]:
        """Returns current tcp position

        Returns:
            Tuple[np.array, np.array]: current position and orientation
        """
        return self.kinematics_solver.compute_forward_kinematics(self.end_effector_name,
                                                                 self.articulation_subset.get_joint_positions())

    def comptute_ik(self, target: Union[list, np.array]) -> Tuple[ArticulationAction, bool]:
        """Computes inverse kinematics for a given target position

        Args:
            target (Union[list, np.array]): list/np.array of position (np.array(3x1))
                and orientation (quaternion np.array(4x1))

        Returns:
            Tuple[ArticulationAction, bool]: An ArticulationAction that can be applied
                to the robot to move the end effector frame to the desired position and
                Solver converged successfully
        """
        return self.ik_object.compute_inverse_kinematics(
            target_position=target[0],
            target_orientation=target[1])

    def compute_fk(self,
                   joint_positions: Union[list, np.array],
                   joint_name: str = "") -> Tuple[np.array, np.array]:
        """Compute the position of a given frame in the robot relative to the USD stage global frame

        Parameters:
            frame_name (str): Name of robot frame on which to calculate forward kinematics
            joint_positions (np.array): Joint positions for the joints returned by get_joint_names()
            position_only (bool): Lula Kinematics ignore this flag and always computes both position and orientation

        Returns:
            frame_positions (3x1): vector describing the translation of the frame relative to the USD stage origin
            frame_rotation (3x3): rotation matrix describing the rotation of the frame relative to the USD stage global frame

        Return type:
            Tuple[np.array, np.array]
        """
        return self.kinematics_solver.compute_forward_kinematics(
            self.end_effector_name,
            joint_positions
        ) if joint_name == "" else self.kinematics_solver.compute_forward_kinematics(
            joint_name,
            joint_positions)

    @property
    def world(self):
        return self.world_object
