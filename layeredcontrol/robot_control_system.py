# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A layered robot control system
"""

import time
from typing import Optional, Dict, Any
import torch
from dataclasses import dataclass
from action_provider.action_base import ActionProvider
from tasks.common_observations.g1_29dof_state import quat_to_rot_matrix
from tools.shared_memory_utils import TransformWriter



@dataclass
class ControlConfig:
    """minimal control configuration"""
    step_hz: int = 500  # the frequency of the low-level execution
    replay_mode: bool = False
    use_rl_action_mode: bool = False


class RobotController:
    """robot controller
    """
    
    def __init__(self, env, config: ControlConfig):
        self.env = env
        self.config = config
        self.action_provider: Optional[ActionProvider] = None
        self.is_running = False
        
        
        # minimal frequency control
        self._step_interval = 1.0 / config.step_hz
        self._last_step_time = 0.0
        
        all_joint_names = env.scene["robot"].data.joint_names
        self._last_action = torch.zeros(len(all_joint_names), device=env.device)
        
        
        # pre-calculate the sleep threshold (avoid calculating every time)
        self._sleep_threshold = 0.0002
        self._sleep_adjustment = 0.0001
        
        # minimal statistics
        self.step_count = 0
        self._start_time = 0.0
        
        # minimal performance analysis
        self._profile_counter = 0
        self._profile_interval = 2000  # reduce the printing frequency
        
        # cache the function reference (reduce the lookup overhead)
        self._perf_counter = time.perf_counter
        self._time_sleep = time.sleep

        # transform shared memory for calibration data collection
        self._transform_writer = TransformWriter()
        body_names = env.scene["robot"].data.body_names
        self._cam_base_body_idx = body_names.index("left_hand_camera_base_link")

        print(f"  - control frequency: {config.step_hz}Hz")
    
    def set_action_provider(self, provider: ActionProvider):
        """set the action provider"""
        if self.action_provider:
            self.action_provider.stop()
            self.action_provider.cleanup()
        
        self.action_provider = provider
        print(f"[SimpleController] set the action provider: {provider.name}")
    
    def start(self):
        """start the controller"""
        if self.is_running:
            return
        
        self.is_running = True
        self._start_time = time.time()
        self._last_step_time = self._perf_counter()
        
        # start the action provider
        if self.action_provider:
            self.action_provider.start()
        
        print("[SimpleController] the controller is started")
    
    def stop(self):
        """stop the controller"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.action_provider:
            self.action_provider.stop()
        
        print("[SimpleController] the controller is stopped")
    
    def step(self):
        """minimal control step - zero thread competition"""
        if not self.is_running:
            return
        
        # use the cached function reference
        perf_counter = self._perf_counter
        step_start = perf_counter()
        
        # 1. minimal action acquisition (synchronous, zero thread competition, pre-calculated strategy)
        action_start = perf_counter()
        action = None
        
        # try to get the action from the action provider
        if self.action_provider:
            action = self.action_provider.get_action(self.env)
            if action is not None:
                self._last_action = action
        
        # if no action is obtained, use the pre-calculated fallback strategy
        if action is None:
            action = self._last_action

        action_time = perf_counter() - action_start
        
        # 2. direct environment step
        env_start = perf_counter()
        with torch.inference_mode():
            if self.config.replay_mode or self.config.use_rl_action_mode:
                pass
                # self.env.sim.render()
            else:
                self.env.step(action)

                # write world→cam_base to shared memory every step (for snap_left_wrist.py)
                try:
                    body_pose_w = self.env.scene["robot"].data.body_link_pose_w
                    p_cb = body_pose_w[0, self._cam_base_body_idx, :3]
                    q_cb = body_pose_w[0, self._cam_base_body_idx, 3:7]
                    R_w_cb = quat_to_rot_matrix(q_cb.unsqueeze(0))[0]
                    self._transform_writer.write(
                        int(time.time() * 1000),
                        p_cb.cpu().numpy(),
                        R_w_cb.cpu().numpy()
                    )
                except Exception:
                    pass

                ### ground truth camera→tag transform ###
                if self.step_count % 60 == 0:
                    try:
                        # Get left wrist camera world pose (ROS convention: +Z forward, -Y up)
                        cam = self.env.scene["left_wrist_camera"]
                        p_cam = cam.data.pos_w[0]           # [3]
                        q_cam = cam.data.quat_w_ros[0]      # [4] (w,x,y,z)

                        # Get AprilTag world pose from scene (XFormPrim)
                        apriltag_xform = self.env.scene["apriltag"]
                        tag_pos_np, tag_quat_np = apriltag_xform.get_world_poses()
                        p_tag = torch.tensor(tag_pos_np[0], device=p_cam.device, dtype=p_cam.dtype)
                        q_tag = torch.tensor(tag_quat_np[0], device=p_cam.device, dtype=p_cam.dtype)

                        # Build rotation matrices
                        R_w_cam = quat_to_rot_matrix(q_cam.unsqueeze(0))[0]  # [3,3]
                        R_w_tag = quat_to_rot_matrix(q_tag.unsqueeze(0))[0]  # [3,3]

                        # T_camera_tag = T_world_camera⁻¹ * T_world_tag
                        R_cam_tag = R_w_cam.T @ R_w_tag
                        t_cam_tag = R_w_cam.T @ (p_tag - p_cam)

                        print(f"\n[GT] camera→tag translation: {t_cam_tag.cpu().numpy()}")
                        print(f"[GT] camera→tag rotation:\n{R_cam_tag.cpu().numpy()}")
                    except Exception as e:
                        print(f"[GT] transform extraction error: {e}")

                    # world → left_hand_camera_base_link
                    try:
                        robot_data = self.env.scene["robot"].data
                        body_names = robot_data.body_names
                        cb_idx = body_names.index("left_hand_camera_base_link")
                        body_pose_w = robot_data.body_link_pose_w  # [B, N, 7]
                        p_cb = body_pose_w[0, cb_idx, :3]
                        q_cb = body_pose_w[0, cb_idx, 3:7]        # (w,x,y,z)
                        R_w_cb = quat_to_rot_matrix(q_cb.unsqueeze(0))[0]
                        print(f"[GT] world→cam_base translation: {p_cb.cpu().numpy()}")
                        print(f"[GT] world→cam_base rotation:\n{R_w_cb.cpu().numpy()}")
                    except Exception as e:
                        print(f"[GT] cam_base transform error: {e}")

                    # solved: left_hand_camera_base_link → left_wrist_camera
                    try:
                        R_cb_cam = R_w_cb.T @ R_w_cam
                        t_cb_cam = R_w_cb.T @ (p_cam - p_cb)
                        print(f"[GT] cam_base→camera translation: {t_cb_cam.cpu().numpy()}")
                        print(f"[GT] cam_base→camera rotation:\n{R_cb_cam.cpu().numpy()}")
                    except Exception as e:
                        print(f"[GT] cam_base→camera solve error: {e}")
                #### end




            env_time = perf_counter() - env_start
            
            self.step_count += 1
        
        # 3. minimal frequency control (no rendering overhead, use the pre-calculated threshold)
        sleep_start = perf_counter()
        current_time = perf_counter()
        if self._last_step_time > 0:
            elapsed = current_time - self._last_step_time
            sleep_needed = self._step_interval - elapsed
            if sleep_needed > self._sleep_threshold:  # use the pre-calculated threshold
                self._time_sleep(sleep_needed - self._sleep_adjustment)  # use the pre-calculated adjustment value
        self._last_step_time = current_time
        sleep_time = perf_counter() - sleep_start
        
        # 4. minimal performance print
        self._profile_counter += 1
        if self._profile_counter >= self._profile_interval:
            total_time = perf_counter() - step_start
            print(f"[Performance] A:{action_time*1000:.1f}ms, E:{env_time*1000:.1f}ms, S:{sleep_time*1000:.1f}ms, T:{total_time*1000:.1f}ms")
            self._profile_counter = 0
    def cleanup(self):
        """clean up the resources"""
        self.stop()
        if self.action_provider:
            self.action_provider.cleanup()
        self._transform_writer.close()
    
    def set_profiling(self, enabled: bool, interval: int = 2000):
        """set the performance analysis"""
        if enabled:
            self._profile_interval = interval
        else:
            self._profile_interval = 999999999  # actually disable the printing
