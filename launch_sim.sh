#!/usr/bin/env bash
# Launch the Unitree Isaac Lab simulation.
#
# Runs sim_main.py inside the `unitree` conda env via the lane wrapper.
# All settings have defaults; override via env vars, and any extra args
# are forwarded straight to sim_main.py.
#
# Examples:
#   ./launch_sim.sh                                   # AprilTag calibration (this branch), Dex3, cuda
#   DEVICE=cpu ./launch_sim.sh                        # run on CPU
#   TASK=Isaac-Stack-RgyBlock-G129-Dex1-Joint HAND=dex1 ./launch_sim.sh
#   ROBOT=h1_2 HAND=inspire TASK=Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint ./launch_sim.sh
#   ./launch_sim.sh --headless --seed 7              # extra flags passed through to sim_main.py
#
# Overridable env vars (with defaults):
#   CONDA_ENV=unitree   DEVICE=cuda   ROBOT=g129
#   HAND=dex3           (dex1 | dex3 | inspire | none)
#   TASK=Isaac-AprilTag-Calibration-G129-Dex3-Joint
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="${CONDA_ENV:-unitree}"
DEVICE="${DEVICE:-cuda}"
TASK="${TASK:-Isaac-PickPlace-RedBlock-G129-Dex3-Joint}"
ROBOT="${ROBOT:-g129}"
HAND="${HAND:-dex3}"

case "$HAND" in
  dex1)    HAND_FLAG="--enable_dex1_dds" ;;
  dex3)    HAND_FLAG="--enable_dex3_dds" ;;
  inspire) HAND_FLAG="--enable_inspire_dds" ;;
  none)    HAND_FLAG="" ;;
  *) echo "ERROR: unknown HAND='$HAND' (use dex1|dex3|inspire|none)" >&2; exit 1 ;;
esac

CMD="python sim_main.py --device ${DEVICE} --enable_cameras --task ${TASK} --robot_type ${ROBOT} ${HAND_FLAG} $*"

echo "[launch_sim] env  : ${CONDA_ENV}"
echo "[launch_sim] dir  : ${REPO_DIR}"
echo "[launch_sim] task : ${TASK}  (robot=${ROBOT}, hand=${HAND}, device=${DEVICE})"
echo "[launch_sim] cmd  : ${CMD}"

# use_conda lives in ~/.bashrc, so run through an interactive shell (-i).
exec bash -ic "use_conda ${CONDA_ENV} && cd '${REPO_DIR}' && ${CMD}"
