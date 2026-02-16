#!/usr/bin/env python3
# Inspect isaac_sim_state shared memory for available robot pose data.

from typing import Any, Dict
from dds.sharedmemorymanager import SharedMemoryManager


def _print_keys(d: Dict[str, Any], prefix: str = ""):
    for k in d.keys():
        print(f"{prefix}{k}")


def main():
    shm = SharedMemoryManager(name="isaac_sim_state", size=4096)
    data = shm.read_data()
    if data is None:
        print("No sim_state data available.")
        return

    print("Top-level keys:")
    _print_keys(data, "  ")

    init_state = data.get("init_state")
    if isinstance(init_state, dict):
        print("\ninit_state keys:")
        _print_keys(init_state, "  ")

        robot = init_state.get("robot")
        if isinstance(robot, dict):
            print("\nrobot keys:")
            _print_keys(robot, "  ")
            body_names = robot.get("body_names")
            body_pose = robot.get("body_link_pose_w")
            if isinstance(body_names, list):
                print(f"\nbody_names: {len(body_names)} entries")
                print(f"  first 5: {body_names[:5]}")
            if body_pose is not None:
                try:
                    import numpy as np
                    arr = np.array(body_pose)
                    print(f"body_link_pose_w shape: {arr.shape}")
                except Exception:
                    print("body_link_pose_w present (shape unavailable)")

    # Fallback direct keys
    if "body_names" in data or "body_link_pose_w" in data:
        print("\nDirect body data detected at top-level:")
        if "body_names" in data and isinstance(data["body_names"], list):
            print(f"body_names: {len(data['body_names'])} entries")
        if "body_link_pose_w" in data:
            try:
                import numpy as np
                arr = np.array(data["body_link_pose_w"])
                print(f"body_link_pose_w shape: {arr.shape}")
            except Exception:
                print("body_link_pose_w present (shape unavailable)")


if __name__ == "__main__":
    main()
