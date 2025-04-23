#!/usr/bin/env python3
import os
import subprocess

# 1) Enable experimental Windows/macOS clustering for this process only
os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"

# 2) Build the command to start the Ray head
cmd = [
    "ray", "start",
    "--head",
    "--port", "6379",
    "--ray-client-server-port", "10001",
    "--dashboard-host", "127.0.0.1",
    "--num-gpus", "1",
    "--block"  # keep Ray running in this terminal
]

# 3) Launch it
subprocess.run(cmd, check=True)
