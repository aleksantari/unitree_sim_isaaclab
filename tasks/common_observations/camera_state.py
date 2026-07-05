# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")


_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 2,
}


_return_placeholder = None
_async_queue = None
_async_thread = None
_async_started = False

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


# --- head depth ZMQ publisher (feeds the robot-side cuRobo Mapper/ESDF collision world) ----
# Decoupled from the JPEG color SHM/ImageServer path: a dedicated PUB socket streams the head
# camera's perpendicular depth as raw float32 MILLIMETERS (Isaac renders meters -> x1000), so
# the robot client treats sim depth byte-identically to the real ZED. Port via env
# ISAAC_HEAD_DEPTH_PORT (default 55556 = color 55555 + 1). Bound lazily; disabled after any
# real failure so a publish hiccup never stalls the sim observation loop.
_depth_pub = None
_depth_pub_off = False
_depth_pub_port = int(os.environ.get("ISAAC_HEAD_DEPTH_PORT", "55556"))


def _publish_head_depth(depth_m) -> None:
    global _depth_pub, _depth_pub_off
    if _depth_pub_off:
        return
    try:
        import numpy as np
        import zmq
        if _depth_pub is None:
            sock = zmq.Context.instance().socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, 1)        # keep only the latest frame
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(f"tcp://0.0.0.0:{_depth_pub_port}")
            _depth_pub = sock
            print(f"[camera_state] head depth PUB on tcp://0.0.0.0:{_depth_pub_port} (float32 mm)")
        d = np.squeeze(np.asarray(depth_m)).astype(np.float32) * 1000.0   # meters -> mm
        d[~np.isfinite(d)] = np.nan                  # inf/no-hit -> NaN (invalid, ZED-like)
        d[d > 1.0e7] = np.nan                        # far-clip background (>10 km) -> invalid
        try:
            _depth_pub.send(np.ascontiguousarray(d, dtype=np.float32).tobytes(), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass                                     # no subscriber / HWM -> drop (normal)
    except Exception as e:                           # never break the sim loop on depth
        _depth_pub_off = True
        print(f"[camera_state] head depth disabled: {e}")


def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """
    global _return_placeholder
    if _return_placeholder is None:
        _return_placeholder = torch.zeros((1, 480, 640, 3))


    _camera_cache['frame_step'] = (_camera_cache['frame_step'] + 1) % max(1, _camera_cache['write_interval_steps'])


    scene_id = id(env.scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(env.scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        _camera_cache['last_scene_id'] = scene_id


    if _camera_cache['frame_step'] == 0:
        try:
            dt = getattr(env, 'physics_dt', 0.02)
            if hasattr(env.scene, 'sensors') and env.scene.sensors:
                for sensor in env.scene.sensors.values():
                    try:
                        sensor.update(dt, force_recompute=False)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # get the camera images
    images = {}
    # env.sim.render()
    

    camera_keys = _camera_cache['camera_keys']
    # Head camera (front camera)
    if "front_camera" in camera_keys:
        head_image = env.scene["front_camera"].data.output["rgb"][0]  # [batch, height, width, 3]

        if head_image.device.type == 'cpu':
            images["head"] = head_image.numpy()
        else:
            images["head"] = head_image.cpu().numpy()

        # head depth -> dedicated ZMQ PUB (robot-side cuRobo collision world), at the SHM
        # color cadence (frame_step==0). float32 mm, decoupled from the JPEG color path.
        if _camera_cache['frame_step'] == 0:
            _fc_out = env.scene["front_camera"].data.output
            if "distance_to_image_plane" in _fc_out:
                _d = _fc_out["distance_to_image_plane"][0]
                _d = _d.numpy() if _d.device.type == 'cpu' else _d.detach().cpu().numpy()
                _publish_head_depth(_d)

    # Left camera (left wrist camera)
    if "left_wrist_camera" in camera_keys:
        left_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
        if left_image.device.type == 'cpu':
            images["left"] = left_image.numpy()
        else:
            images["left"] = left_image.cpu().numpy()
    
    # Right camera (right wrist camera)  
    if "right_wrist_camera" in camera_keys:
        right_image = env.scene["right_wrist_camera"].data.output["rgb"][0]
        if right_image.device.type == 'cpu':
            images["right"] = right_image.numpy()
        else:
            images["right"] = right_image.cpu().numpy()
    
    # if no camera with the specified name is found, try other common camera names
    if not images:

        available_cameras = _camera_cache['available_cameras']
        if available_cameras:
            print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
            
            # if there are available cameras, use the first three as head, left, right
            for i, camera_name in enumerate(available_cameras[:3]):
                camera_image = env.scene[camera_name].data.output["rgb"][0]
                
               
                if camera_image.device.type == 'cpu':
                    numpy_image = camera_image.numpy()
                else:
                    numpy_image = camera_image.cpu().numpy()
                
                if i == 0:
                    images["head"] = numpy_image
                elif i == 1:
                    images["left"] = numpy_image
                elif i == 2:
                    images["right"] = numpy_image
    

    if images and _camera_cache['frame_step'] == 0:
        _ensure_async_started()
        try:
            
            if _async_queue.full():
                _async_queue.get_nowait()
            _async_queue.put_nowait(images)
        except Exception:
            pass
    elif not images:
        print("[camera_state] No camera images found in the environment")
    
    return _return_placeholder

