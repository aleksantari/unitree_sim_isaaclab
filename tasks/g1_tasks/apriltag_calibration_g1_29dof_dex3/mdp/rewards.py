# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import torch
from isaaclab.envs import ManagerBasedRLEnv


def compute_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dummy reward for calibration task (always 0)."""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float)


__all__ = ["compute_reward"]
