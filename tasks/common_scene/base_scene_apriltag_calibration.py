# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Scene configuration for AprilTag calibration task.
Static AprilTag block on the packing table for eye-in-hand camera calibration.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")

@configclass
class TableAprilTagCalibrationSceneCfg(InteractiveSceneCfg):
    """Scene with a static AprilTag block on the packing table.
    Used for eye-in-hand camera calibration via teleoperation.
    """

    # 1. room wall configuration
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
        ),
    )

    # 2. table configuration
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-4.3, -4.2, -0.2],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",
        ),
    )

    # 3. static AprilTag block (non-interactive, fixed to table)
    apriltag = AssetBaseCfg(
        prim_path="/World/envs/env_.*/AprilTag",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-4.08968, -4.17733, 0.87519],
            rot=[0.7071, -0.7071, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/apriltag_block/apriltag_block.usd",
            scale=(2.0, 2.0, 2.0),  # 5cm USD → 10cm in sim
        ),
    )




    # 4. world camera
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-4.1, -4.9, 1.0),
        rot_offset=(-0.3173, 0.94833, 0.0, 0.0)
    )
