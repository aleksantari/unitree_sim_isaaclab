# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Multi-prop "clutter" pick-place env for the G1 29-DoF + Dex3 (joint control).

Clone of pickplace_redblock_g1_29dof_dex3_joint_env_cfg.py, but the scene is the
multi-prop TablePropsSceneCfg (cube + sphere + cylinder + cone + capsule + mug +
toy_truck) instead of the single red block. Used to stress the off-board GraspGenX
pipeline (SAM3 segmentation among distractors, 6-DoF grasps on non-box geometry,
depth-ESDF collision world with multiple obstacles). The redblock task is left
untouched. `mdp` (observations / terminations / rewards) is reused from the
redblock task; they key on the `object` prop, which this scene keeps.
"""

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

# reuse the redblock task's mdp (observations/terminations/rewards are identical)
from tasks.g1_tasks.pick_place_redblock_g1_29dof_dex3 import mdp

from tasks.common_config import G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# multi-prop clutter scene
from tasks.common_scene.base_scene_pickplace_props import TablePropsSceneCfg, PROP_NAMES

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(TablePropsSceneCfg):
    """Multi-prop table scene + the G1 robot and cameras."""

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071))

    # cameras (front_camera carries the head depth PUB on :55556)
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP (direct joint angle control)."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


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
    # check if the object is out of the working range
    success = DoneTerm(func=mdp.reset_object_estimate)


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # props pinned at their default poses (no randomization) for a stable GT, matching
            # the redblock convention.
            "pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlacePropsG129DEX3BaseFixEnvCfg(ManagerBasedRLEnvCfg):
    """G1 29-DoF + Dex3 multi-prop pick-place environment configuration."""

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1,
                                                     env_spacing=2.5,
                                                     replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 120.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        # more rigid bodies than the single-block scene -> a little more headroom for contact pairs
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 4
        self.sim.physx.contact_offset = 0.01
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 4

        self.event_manager = SimpleEventManager()

        # reset every prop to its fixed default pose (stable GT for the external client)
        self.event_manager.register_multi_object_reset(
            name="reset_object_self",
            object_names=PROP_NAMES,
            pose_ranges={"x": [0.0, 0.0], "y": [0.0, 0.0]},
            velocity_ranges={},
        )
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))
