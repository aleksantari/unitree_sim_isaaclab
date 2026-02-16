#!/usr/bin/env python3
# Snap images on demand from Isaac Sim shared memory (left/right/head).

import argparse
import os
import time
from datetime import datetime

import cv2

from tools.shared_memory_utils import MultiImageReader


def _save_image(img, out_dir: str, prefix: str, ext: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(out_dir, f"{prefix}_{ts}.{ext}")
    cv2.imwrite(path, img)
    return path


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
                    path = _save_image(img, args.out_dir, args.prefix, args.ext)
                    print(f"Saved {path}")
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
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
        path = _save_image(img, args.out_dir, args.prefix, args.ext)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
