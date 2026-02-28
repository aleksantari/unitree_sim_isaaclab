#!/usr/bin/env python3
"""Test script to capture a single image from the videohub DDS service.

Works against both the real Unitree robot and the simulator (when running
sim_main.py with --enable_cameras).

Usage:
    python tools/test_videohub.py                    # auto-detect DDS
    python tools/test_videohub.py wlp13s0            # specify network interface
    python tools/test_videohub.py wlp13s0 --output /tmp/out.jpg
"""

import sys
import argparse
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


def main():
    parser = argparse.ArgumentParser(description="Capture image via videohub DDS")
    parser.add_argument("interface", nargs="?", default=None,
                        help="Network interface (e.g. wlp13s0). Omit for auto-detect.")
    parser.add_argument("--output", default="./videohub_capture.jpg",
                        help="Output JPEG path (default: ./videohub_capture.jpg)")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="RPC timeout in seconds (default: 3.0)")
    args = parser.parse_args()

    # Initialize DDS channel
    if args.interface:
        print(f"Initializing DDS on interface: {args.interface}")
        ChannelFactoryInitialize(1, args.interface)
    else:
        print("Initializing DDS (auto-detect interface)")
        ChannelFactoryInitialize(1)

    # Connect to videohub service
    client = VideoClient()
    client.SetTimeout(args.timeout)
    client.Init()

    print("Calling GetImageSample()...")
    code, data = client.GetImageSample()

    if code != 0:
        print(f"FAILED: GetImageSample returned error code {code}")
        sys.exit(1)

    if not data:
        print("FAILED: GetImageSample returned code 0 but empty data")
        print("  -> Cameras may not be initialized. Did you pass --enable_cameras to sim_main.py?")
        sys.exit(1)

    # Decode JPEG and validate
    import cv2
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if img is None:
        print(f"FAILED: Received {len(data)} bytes but cv2.imdecode returned None (not a valid JPEG)")
        sys.exit(1)

    h, w, c = img.shape
    print(f"SUCCESS: Received valid image — {w}x{h} px, {len(data)} bytes")

    cv2.imwrite(args.output, img)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
