#!/usr/bin/env python3
# Snap images on demand from Isaac Sim shared memory (left/right/head).
# Also saves the corresponding world→cam_base transform for calibration.

import argparse
import json
import os
import time
from datetime import datetime

import cv2

from tools.shared_memory_utils import MultiImageReader, TransformReader


def _save_capture(img, transform_data, out_dir: str, prefix: str, ext: str):
    """Save image and transform with matching timestamps."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{prefix}_{ts}"

    img_path = os.path.join(out_dir, f"{base}.{ext}")
    cv2.imwrite(img_path, img)

    if transform_data is not None:
        timestamp_ms, position, rotation_matrix = transform_data
        tf_path = os.path.join(out_dir, f"{base}.json")
        payload = {
            "world_to_cam_base": {
                "position": position.tolist(),
                "rotation_matrix": rotation_matrix.tolist(),
            },
            "timestamp_ms": int(timestamp_ms),
        }
        with open(tf_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved {img_path} + {tf_path}")
    else:
        print(f"Saved {img_path} (no transform available)")


def _wait_for_frame(reader: MultiImageReader, source: str, timeout_s: float):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        img = reader.read_single_image(source)
        if img is not None:
            return img
        time.sleep(0.02)
    return None


def main():
    parser = argparse.ArgumentParser(description="Snap images on demand from Isaac Sim shared memory.")
    parser.add_argument("--source", choices=["left", "right", "head"], default="left",
                        help="which camera source to read (default: left)")
    parser.add_argument("--out_dir", default="snapshots", help="output directory")
    parser.add_argument("--prefix", default="left_wrist", help="filename prefix")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"], help="file extension")
    parser.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for a frame")
    parser.add_argument("--preview", action="store_true",
                        help="show live preview; press 's' to save, 'q' to quit")
    args = parser.parse_args()

    reader = MultiImageReader()
    tf_reader = TransformReader()

    if args.preview:
        print("Preview mode: press 's' to save, 'q' to quit.")
        while True:
            img = reader.read_single_image(args.source)
            if img is not None:
                cv2.imshow(f"{args.source} camera", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                if img is None:
                    img = _wait_for_frame(reader, args.source, args.timeout)
                if img is None:
                    print("No frame available to save.")
                else:
                    _save_capture(img, tf_reader.read(), args.out_dir, args.prefix, args.ext)
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        tf_reader.close()
        return

    print("Press Enter to capture, or type 'q' then Enter to quit.")
    while True:
        cmd = input()
        if cmd.strip().lower() == "q":
            break
        img = _wait_for_frame(reader, args.source, args.timeout)
        if img is None:
            print("No frame available to save.")
            continue
        _save_capture(img, tf_reader.read(), args.out_dir, args.prefix, args.ext)

    tf_reader.close()


if __name__ == "__main__":
    main()
