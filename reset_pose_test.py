#!/usr/bin/env python3
"""
publish reset category command to rt/reset_pose/cmd
"""

import os
import sys
import time
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

def publish_reset_category(category: int,publisher):
    # construct message
    msg = String_(data=str(category))  # pass data parameter directly during initialization

    # create publisher

    # publish message
    publisher.Write(msg)
    print(f"published reset category: {category}")

if __name__ == "__main__":
    # category: 1 = reset object(s), 2 = reset all (non-wholebody only)
    category = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    # Must use the SAME network interface as the sim. The sim calls
    # ChannelFactoryInitialize(1, os.environ["UNITREE_DDS_IFACE"]); the unitree SDK
    # builds its CycloneDDS config inline and IGNORES CYCLONEDDS_URI, so the
    # interface must be matched here too (default "lo" for the loopback setup).
    iface = os.environ.get("UNITREE_DDS_IFACE", "lo")
    print(f"initializing DDS on domain 1, interface '{iface}'")
    ChannelFactoryInitialize(1, iface)
    publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
    publisher.Init()

    # publish repeatedly to survive the discovery race (subscriber needs time
    # to discover this freshly-created participant before the first Write).

    publish_reset_category(category, publisher)
    time.sleep(0.3)
    print("test publish completed")