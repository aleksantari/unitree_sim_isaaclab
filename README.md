# AprilTag Eye-in-Hand Calibration — Branch Guide

**Branch:** `apriltag-calibration-task`
**Base repo:** [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)

This branch adds a complete eye-in-hand camera calibration pipeline to the simulator. It enables you to solve for the fixed transform between the robot's wrist camera (`left_wrist_camera`) and its mounting link (`left_hand_camera_base_link`) using an AprilTag as a calibration target.

---

## Overview

Eye-in-hand calibration solves the classic **AX = XB** problem:
- **A** = relative motion of the robot end-effector in the world frame
- **B** = relative motion of the camera-observed calibration target
- **X** = the unknown fixed camera extrinsic (cam_base → camera)

This branch provides:
1. A static AprilTag USD asset as the calibration target
2. A dedicated IsaacLab task (`Isaac-AprilTag-Calibration-G129-Dex3-Joint`) with the G1 robot
3. A shared-memory data collection pipeline that synchronizes captured images with robot transforms
4. Ground truth logging for validation

---

## Branch Changes

| File | Status | Purpose |
|------|--------|---------|
| `assets/objects/apriltag_block/create_apriltag_block.py` | NEW | Programmatically generates the AprilTag USD asset using the `pxr` API |
| `assets/objects/apriltag_block/apriltag_14.png` | NEW | AprilTag texture (tag36h11, ID 14) copied from warehouse scene assets |
| `assets/objects/apriltag_block/apriltag_block.usd` | NEW | Generated 5cm × 5cm × 3mm thin block USD with AprilTag texture on top face |
| `tasks/common_scene/base_scene_apriltag_calibration.py` | NEW | Scene config: warehouse room + packing table + static AprilTag at 10cm scale |
| `tasks/g1_tasks/apriltag_calibration_g1_29dof_dex3/__init__.py` | NEW | Gym task registration for `Isaac-AprilTag-Calibration-G129-Dex3-Joint` |
| `tasks/g1_tasks/apriltag_calibration_g1_29dof_dex3/apriltag_calibration_g1_29dof_dex3_joint_env_cfg.py` | NEW | Full MDP env config (robot + cameras + 10-min episodes + stub reward) |
| `tasks/g1_tasks/apriltag_calibration_g1_29dof_dex3/mdp/` | NEW | MDP modules (observations, stub zero reward) |
| `tasks/g1_tasks/__init__.py` | MODIFIED | Imports and registers the new task |
| `tools/shared_memory_utils.py` | MODIFIED | Added `TransformWriter` and `TransformReader` classes for IPC |
| `layeredcontrol/robot_control_system.py` | MODIFIED | Publishes world→cam_base transform to shared memory each sim step; logs GT transforms every 60 steps |
| `tools/snap_left_wrist.py` | MODIFIED | Rewritten to save paired `.png` + `.json` (image + world→cam_base transform) on each capture |

---

## Full Pipeline

```
┌─────────────────────────────────────┐
│  Isaac Sim (this repo)              │
│                                     │
│  1. Launch calibration task         │
│  2. Teleop robot arm to ~20+ poses  │
│  3. Press Enter to capture          │
│     → saves image + transform pair  │
└──────────────┬──────────────────────┘
               │ snapshots/*.png
               │ snapshots/*.json
               ▼
┌─────────────────────────────────────┐
│  External detector repo             │
│                                     │
│  4. Batch detect AprilTag in images │
│     → produces cam→tag transforms   │
│  5. Merge with world→cam_base JSONs │
│     → calibration_pairs.json        │
└──────────────┬──────────────────────┘
               │ calibration_pairs.json
               ▼
┌─────────────────────────────────────┐
│  Calibration script                 │
│                                     │
│  6. cv2.calibrateHandEye(...)       │
│     → cam_base→camera extrinsic     │
└─────────────────────────────────────┘
```

---

## Prerequisites

**This repo (isaaclab-unitree conda env):**
- IsaacLab + Isaac Sim installed per the main README
- `xr_teleoperate` package for DDS-based robot teleoperation

**External detector/calibration repo (separate conda env):**
```bash
pip install dt-apriltags opencv-python numpy scipy
```

---

## Step 1: Launch the Calibration Sim

```bash
python sim_main.py \
    --device cuda \
    --enable_cameras \
    --task Isaac-AprilTag-Calibration-G129-Dex3-Joint \
    --enable_dex3_dds \
    --robot_type g129
```

The sim spawns the G1 robot base-fixed on the packing table with a 10cm AprilTag placed in front of it. Episode length is 10 minutes.

> **Viewport tip:** If you cannot zoom close enough in the viewport, click the AprilTag or robot, press `F` to re-focus, then zoom. Alternatively use `Alt + Right-click drag` for true dolly (no pivot limit).

### Ground truth logging

Every 60 simulation steps, the following transforms are printed to the terminal for validation:

```
[GT] camera→tag:       R and t of the AprilTag in the camera frame
[GT] world→cam_base:   Pose of left_hand_camera_base_link in world
[GT] cam_base→camera:  Solved cam_base→camera (ground truth answer)
```

Use these to validate your detector output and final calibration result.

---

## Step 2: Collect the Dataset

In a **second terminal**, run the capture tool:

```bash
# Preview mode (recommended) — see live camera feed, press 's' to capture
python tools/snap_left_wrist.py --preview --out_dir snapshots --prefix left_wrist

# CLI mode — press Enter to capture, 'q' + Enter to quit
python tools/snap_left_wrist.py --out_dir snapshots --prefix left_wrist
```

**Collection workflow:**
1. Use `xr_teleoperate` to move the wrist camera to a pose where the AprilTag is clearly visible
2. **Stop moving** and let the robot settle (~1 second)
3. Press `s` (preview) or Enter (CLI) to capture
4. Move to a new pose — vary position, distance, and viewing angle significantly
5. Collect **at least 20 poses** for robust calibration (more is better)

**Output per capture:**
```
snapshots/left_wrist_YYYYMMDD_HHMMSS_ffffff.png   ← 640×480 RGB camera image
snapshots/left_wrist_YYYYMMDD_HHMMSS_ffffff.json  ← world→cam_base transform
```

**JSON format:**
```json
{
  "world_to_cam_base": {
    "position": [x, y, z],
    "rotation_matrix": [[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]]
  },
  "timestamp_ms": 1234567890
}
```

> The `.png` and `.json` files share the same timestamp prefix — this is what links them as a matched pair.

---

## Step 3: Run AprilTag Detection (External Repo)

In your external detector environment, batch-process all captured images to produce `cam→tag` transforms. Key parameters:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `tag_family` | `"tag36h11"` | Tag family used in this branch |
| `tag_size` | `0.08` | **Metres.** 10cm physical tag × 0.8 = 8cm (outer black border boundary — dt_apriltags convention) |
| Rotation fix | `R_fix = np.diag([1, -1, -1])` | Applied as `R = detection.pose_R @ R_fix` (post-multiply) |

**Why the rotation fix?** The `dt_apriltags` library uses a different tag-frame convention than IsaacLab. The fix corrects a 180° rotation around the tag's X-axis. Translation (`detection.pose_t`) does **not** need correction.

**Output format** (`calibration_pairs.json`):
```json
[
  {
    "image_file": "left_wrist_20260217_131204_028446.png",
    "cam_to_tag": {
      "rotation_matrix": [[...], [...], [...]],
      "translation": [x, y, z]
    },
    "world_to_cam_base": {
      "position": [x, y, z],
      "rotation_matrix": [[...], [...], [...]]
    }
  },
  ...
]
```

---

## Step 4: Run Eye-in-Hand Calibration (External Repo)

Use `cv2.calibrateHandEye()` with the following parameter mapping:

```python
import cv2
import numpy as np

# Build input lists from calibration_pairs.json
R_gripper2base = []   # list of 3x3 np.float64 arrays
t_gripper2base = []   # list of (3,1) np.float64 arrays
R_target2cam   = []   # list of 3x3 np.float64 arrays
t_target2cam   = []   # list of (3,1) np.float64 arrays

for pair in pairs:
    R_gripper2base.append(np.array(pair["world_to_cam_base"]["rotation_matrix"], dtype=np.float64))
    t_gripper2base.append(np.array(pair["world_to_cam_base"]["position"], dtype=np.float64).reshape(3,1))
    R_target2cam.append(np.array(pair["cam_to_tag"]["rotation_matrix"], dtype=np.float64))
    t_target2cam.append(np.array(pair["cam_to_tag"]["translation"], dtype=np.float64).reshape(3,1))

R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam,   t_target2cam,
    method=cv2.CALIB_HAND_EYE_PARK   # PARK or DANIILIDIS recommended
)
```

**Parameter mapping explained:**
- `R_gripper2base` / `t_gripper2base` = world→cam_base (the cam_base pose in the world frame)
- `R_target2cam` / `t_target2cam` = cam→tag (the AprilTag pose in the camera frame)
- **Output:** `R_cam2gripper`, `t_cam2gripper` = the solved cam_base→camera extrinsic

**Expected results** (validated in simulation):

| Method | Rotation Error | Translation Error |
|--------|---------------|-------------------|
| TSAI   | ~2.1°         | ~6.6 mm           |
| PARK   | ~1.9°         | ~6.3 mm           |
| HORAUD | ~1.9°         | ~6.3 mm           |
| DANIILIDIS | ~1.9°    | ~6.4 mm           |

Errors measured against simulator ground truth. PARK and HORAUD are recommended. ~2° / ~6mm is consistent with monocular AprilTag pose estimation accuracy at 640×480 resolution — not a sign of pipeline error.

**Validation:** Compare against the `[GT] cam_base→camera` values printed by the sim during data collection.

---

## Technical Notes

### Shared Memory IPC Architecture

The capture pipeline uses POSIX shared memory to transfer data between the Isaac Sim process and the capture tool process without blocking either side:

| Channel | SHM Name | Write Rate | Content |
|---------|----------|-----------|---------|
| Images | `isaac_left_image_shm` | ~100 Hz | Header (48B) + raw BGR or JPEG payload |
| Transform | `isaac_cam_base_transform_shm` | ~500 Hz | `<Q3d9d` binary: uint64 ts + 3×float64 pos + 9×float64 rot |

The capture tool reads the latest value from both channels at the moment of capture. They are **not frame-locked** — there is a potential temporal skew of up to ~12ms. At human teleoperation speeds with a stationary pause before each capture, this skew is negligible for calibration purposes.

### Coordinate Frame Conventions

- **IsaacLab camera pose:** Uses `quat_w_ros` convention — ROS camera frame (+Z forward, +X right, -Y up)
- **dt_apriltags output:** Same ROS/OpenCV camera convention — no frame conversion needed for translation
- **Rotation fix:** `R_fix = np.diag([1, -1, -1])` corrects the tag-frame axis convention difference between IsaacLab and the AprilTag C library (180° around tag X-axis)
- **Quaternion format:** `body_link_pose_w` returns `[w, x, y, z]` wxyz order

### tag_size Convention (dt_apriltags)

`tag_size` = the distance across the **outer edge of the black border** (excluding the white quiet zone).

For tag36h11, the total tag is 10×10 cells. The black border spans 8×8 cells:

```
tag_size = full_physical_width × (8/10) = 0.10 × 0.8 = 0.08 m
```

The USD is authored at 5cm (0.05m) and scaled 2× in the scene config → 10cm physical size → `tag_size = 0.08`.

---

## Regenerating the AprilTag USD

If you need to change the texture or block dimensions, edit `create_apriltag_block.py` and re-run:

```bash
conda run -n isaaclab-unitree python assets/objects/apriltag_block/create_apriltag_block.py
```

This overwrites `apriltag_block.usd` and re-copies the texture. The `scale=(2.0, 2.0, 2.0)` in `base_scene_apriltag_calibration.py` applies on top at runtime — if you double the USD dimensions, remove the 2× scale and update `tag_size` accordingly.

**Current AprilTag size flow:**
```
USD authored:  5cm × 5cm × 3mm
Scene scale:   2.0 × 2.0 × 2.0
Physical sim:  10cm × 10cm × 6mm
tag_size:      0.08 m  (= 0.10 × 0.8, black border only)
```
