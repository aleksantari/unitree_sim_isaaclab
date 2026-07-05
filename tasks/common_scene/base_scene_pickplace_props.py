# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Multi-prop "clutter" pick-place base scene.

A robustness-testing scene for the off-board GraspGenX pipeline: instead of the
single red cube, it places several geometrically-distinct props on the table at
once (cube + sphere + cylinder + cone + capsule, plus any downloaded USD props).
Lets the head-camera -> SAM3 -> GraspGenX -> cuRobo grasp path be exercised on
non-box geometry and among distractors / collision-world obstacles, all in sim.

Mirrors base_scene_stack_rgyblock.py (room + table + dome light + world camera).
One prop keeps the name/prim `object` (red cube) so the existing `rt/sim_state`
`object_key`, the sim_cloud GT-cube path, and the redblock termination/reward
(both keyed to "object") all keep working unchanged.

Table top is ~z=0.81 (the redblock 6 cm cube sits at center z=0.84). Each prop's
init z = table_top + height/2 (+ a small epsilon); gravity settles the rest.
Poses are a loose cluster within the head-cam FOV + arm reach -- eyeball/tune in
the viewport. Props are procedural primitives (sim_utils.*Cfg) so they need no
external assets; USD props (if downloaded into assets/objects/) are appended below.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")

# Which prop row(s) to spawn -- toggle at launch with `PROPS_ROW=front|back|both` (default both),
# e.g. `PROPS_ROW=front TASK=Isaac-PickPlace-Props-G129-Dex3-Joint ./launch_sim.sh`. The red cube
# `object` is ALWAYS spawned (the env's termination/reward managers + the sim_cloud path reference
# it by name -- removing it crashes env build); PROPS_ROW toggles the OTHER props by row.
_ROW = os.environ.get("PROPS_ROW", "both").strip().lower()
if _ROW not in ("front", "back", "both"):
    _ROW = "both"
_FRONT = _ROW in ("front", "both")
_BACK = _ROW in ("back", "both")


def _shape(geom, color):
    """Attach the shared (redblock-matched) rigid/mass/collision/physics props and a
    diffuse color to a geometry spawn cfg (CuboidCfg/SphereCfg/...). Fresh sub-cfg
    instances per call (no shared mutable defaults)."""
    geom.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False, retain_accelerations=False)
    geom.mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
    geom.collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=0.01, rest_offset=0.0)
    geom.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0)
    geom.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max", restitution_combine_mode="min",
        static_friction=10, dynamic_friction=1.5, restitution=0.01)
    return geom


def _prop(name, pos, geom, rot=(1.0, 0.0, 0.0, 0.0)):
    """A RigidObjectCfg for a single prop at `pos` (pelvis-world xyz), orientation `rot`
    (wxyz quat), with spawn `geom`."""
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(pos), rot=list(rot)),
        spawn=geom,
    )


def _usd_prop(name, pos, usd_rel, scale=(1.0, 1.0, 1.0), mass=None, rot=(1.0, 0.0, 0.0, 0.0)):
    """RigidObjectCfg for a downloaded USD prop under assets/objects/<usd_rel>. Collision
    geometry + materials come from the USD itself (these are Isaac manipulation assets); we
    only add the rigid-body solver props (and optionally override mass). rigid_props match the
    IsaacLab agibot place tasks. Spawn slightly above the table -- gravity settles it onto the
    surface regardless of the asset's authored origin."""
    spawn = UsdFileCfg(
        usd_path=f"{project_root}/assets/objects/{usd_rel}",
        scale=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0, max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0, disable_gravity=False),
    )
    if mass is not None:
        spawn.mass_props = sim_utils.MassPropertiesCfg(mass=mass)
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(pos), rot=list(rot)),
        spawn=spawn,
    )


@configclass
class TablePropsSceneCfg(InteractiveSceneCfg):
    """Table scene with multiple graspable props for grasp-pipeline robustness tests."""

    # 1. room
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
        ),
    )

    # 2. table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",
        ),
    )

    # 3. props (table top ~z=0.81). 3 columns x 2 rows -- front row y=-4.03, back row y=-4.15.
    #    Toggle which row spawns with the PROPS_ROW env var (front|back|both, default both).
    #    Columns in world x (robot faces -y, so its right is -x, its left is +x):
    #      -4.20 : `object` -- now a CYLINDER (was the red cube; kept named `object`, ALWAYS on,
    #              since the managers/sim_cloud reference it). No back-row prop in this column.
    #      -4.50 : to the robot's RIGHT (toy_truck front, brick back).
    #      -3.90 : to the robot's LEFT (sphere front, mug back).
    #    z is each object's own rest height; taller/USD props start a little high and settle.
    object = _prop("Object", (-4.20, -4.03, 0.86),
                   _shape(sim_utils.CylinderCfg(radius=0.025, height=0.10), (0.1, 0.8, 0.2)))
    if _FRONT:   # front row (y=-4.03)
        sphere_prop = _prop("Sphere", (-3.90, -4.03, 0.855),
                            _shape(sim_utils.SphereCfg(radius=0.045), (0.1, 0.25, 0.9)))
        toy_truck_prop = _usd_prop("ToyTruck", (-4.50, -4.03, 0.92),
                                   "toy_truck/toy_truck.usd", mass=0.05,
                                   scale=(1.2, 1.2, 1.2),
                                   rot=(0.9239, 0.0, 0.0, -0.3827))  # -45 deg about +Z, +20% size
    if _BACK:    # back row (y=-4.15)
        mug_prop = _usd_prop("Mug", (-3.90, -4.15, 0.90), "mug/mug.usd",
                             scale=(0.7, 0.7, 0.7))  # shrink 30%
        brick_prop = _prop("Brick", (-4.50, -4.15, 0.85),
                           _shape(sim_utils.CuboidCfg(size=(0.104, 0.052, 0.065)), (0.95, 0.5, 0.1)),
                           rot=(0.9239, 0.0, 0.0, -0.3827))  # -45 deg about +Z, +30% size

    # 4. light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0),
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                   pos_offset=(-4.1, -4.9, 1.8),
                                                   rot_offset=(-0.3173, 0.94833, 0.0, 0.0))


# Names of the ACTIVE prop RigidObjectCfgs (respecting PROPS_ROW) -- used by the env cfg's
# multi-object reset. `object` is always present; the rest depend on PROPS_ROW.
PROP_NAMES = ["object"]
if _FRONT:
    PROP_NAMES += ["sphere_prop", "toy_truck_prop"]
if _BACK:
    PROP_NAMES += ["mug_prop", "brick_prop"]
