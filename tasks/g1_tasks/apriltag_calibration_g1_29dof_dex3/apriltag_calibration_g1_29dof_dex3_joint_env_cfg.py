# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets  # isort: skip

# import scene configuration
from tasks.common_scene.base_scene_apriltag_calibration import TableAprilTagCalibrationSceneCfg

##
# Scene definition
##

@configclass
class AprilTagCalibrationSceneCfg(TableAprilTagCalibrationSceneCfg):
    """Scene with G1 robot and static AprilTag block for calibration."""

    # humanoid robot
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071)
    )

    # cameras
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Time-based termination only (long episode for teleoperation)."""
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Dummy reward (always 0)."""
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class AprilTagCalibrationG129DEX3EnvCfg(ManagerBasedRLEnvCfg):
    """AprilTag calibration environment for G1 29-DOF Dex3."""

    # scene
    scene: AprilTagCalibrationSceneCfg = AprilTagCalibrationSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    # no events, commands, or curriculum needed
    events = None
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 600.0  # 10 minutes for teleoperation
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 4
        self.sim.physx.contact_offset = 0.01
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 4
