# unitree_sim_isaaclab — `upgrades` branch

**Base repo:** [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
**Branch role:** primary development line for the G1 sim→real manipulation stack.

This branch carries the sim-integration work layered on top of Unitree's Isaac Lab
simulator so that a simulated **G1-29dof + Dex3** robot streams data to the real-robot
control stack (`G1_classical_manip`, cuRobo) byte-identically to the physical setup. It
lets the manipulation pipeline be developed and tested entirely in sim before touching
hardware.

## What this branch adds

- **Multi-prop pick-place task** (`Isaac-PickPlace-Props-G129-Dex3-Joint`) — a scene with
  multiple graspable props arranged in rows, selectable at launch via `PROPS_ROW=front|back|both`.
- **Head-camera depth over ZMQ** — the head front camera renders `distance_to_image_plane`
  alongside RGB and publishes it as raw `float32` millimeters on a dedicated PUB socket
  (`ISAAC_HEAD_DEPTH_PORT`, default `55556`), so the robot-side **cuRobo** Mapper/ESDF
  collision world consumes sim depth exactly as it would the real ZED head.
- **AprilTag-on-block scenes** and Dex3 red-block sim-integration support.
- **Larger sim-state DDS buffer** (4 KiB → 32 KiB) so multi-object scenes don't overflow
  the shared-memory serialization and stall `rt/sim_state`.

## Launch

The sim runs via [`launch_sim.sh`](launch_sim.sh) (all settings have env-var defaults; extra
args pass through to `sim_main.py`). The canonical multi-prop launch, wired for a **loopback
DDS** hand-off to the local control stack:

```bash
UNITREE_DDS_IFACE=lo \
CYCLONEDDS_URI=file://$HOME/repos/G1_classical_manip/configs/cyclonedds_loopback.xml \
PROPS_ROW=front \
TASK=Isaac-PickPlace-Props-G129-Dex3-Joint \
./launch_sim.sh
```

- `UNITREE_DDS_IFACE=lo` — bind DDS to loopback (sim and control stack on one machine). The
  client **must** use the same interface; the Unitree SDK ignores `CYCLONEDDS_URI` for
  interface selection, so it is passed explicitly here.
- `CYCLONEDDS_URI=…/cyclonedds_loopback.xml` — CycloneDDS config for the loopback transport.
- `PROPS_ROW=front` — spawn only the front row of props.
- `TASK=Isaac-PickPlace-Props-G129-Dex3-Joint` — the multi-prop Dex3 task.

All DDS traffic uses **channel/domain `1`** — any other DDS instance must initialize with
`ChannelFactoryInitialize(1, <iface>)` on the matching interface to exchange messages.

## Notes

- The sim runs inside the `unitree` conda env (invoked through the lane wrapper by
  `launch_sim.sh`).
- The prior AprilTag eye-in-hand **calibration** guide lives on the `dev` and
  `apriltag-calibration-task` branches.
