import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ctypes
import cv2
import numpy as np
from multiprocessing import shared_memory as shm_module

from unitree_sdk2py.rpc.server import Server
from unitree_sdk2py.go2.video.video_api import (
    VIDEO_SERVICE_NAME,
    VIDEO_API_VERSION,
    VIDEO_API_ID_GETIMAGESAMPLE,
)
from tools.shared_memory_utils import get_shm_name, SimpleImageHeader


class VideoServerDDS(Server):
    """DDS RPC server that mirrors the real robot's 'videohub' service.

    Clients using VideoClient.GetImageSample() will work identically
    against this server as they do against the real robot.
    """

    def __init__(self, jpeg_quality: int = 85):
        super().__init__(VIDEO_SERVICE_NAME)  # "videohub"
        self._jpeg_quality = jpeg_quality
        self._head_shm = None  # lazily opened on first request

    def Init(self):
        self._SetApiVersion(VIDEO_API_VERSION)
        self._RegistBinaryHandler(
            VIDEO_API_ID_GETIMAGESAMPLE, self._GetImageSample, False
        )

    def _GetImageSample(self, parameter: list) -> tuple:
        try:
            shm_name = get_shm_name('head')

            # Lazily open the shared memory segment
            if self._head_shm is None:
                self._head_shm = shm_module.SharedMemory(name=shm_name)

            shm = self._head_shm
            header_size = ctypes.sizeof(SimpleImageHeader)

            header_data = bytes(shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            if header.data_size == 0:
                return 1, []

            payload = bytes(shm.buf[header_size:header_size + header.data_size])

            if header.encoding == 1:  # already JPEG — return directly
                return 0, list(payload)

            # Raw BGR — encode to JPEG
            img = np.frombuffer(payload, dtype=np.uint8).reshape(
                header.height, header.width, header.channels
            )
            ok, buf = cv2.imencode(
                '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                return 1, []
            return 0, list(buf.tobytes())

        except FileNotFoundError:
            # Shared memory not created yet (cameras not initialized)
            return 1, []
        except Exception:
            # Re-open SHM next call in case it was recreated
            self._head_shm = None
            return 1, []
