#!/usr/bin/env python3
import os
import subprocess

#
# ========== CONFIGURE AT THE TOP ==========
#
# Enable Windows/macOS multi-node cluster support?
ENABLE_CLUSTER = True

# Ray head ports
HEAD_PORT          = 6379        # GCS metadata
CLIENT_SERVER_PORT = 10001       # ray client
DASHBOARD_HOST     = "127.0.0.1" # bind dashboard only locally

# Resource declaration
NUM_GPUS           = 1           # tell Ray how many GPUs this node has
# =========================================

if ENABLE_CLUSTER:
    os.environ["RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER"] = "1"

# Build the CLI arguments
cmd = [
    "ray", "start",
    "--head",
    "--port", str(HEAD_PORT),
    "--ray-client-server-port", str(CLIENT_SERVER_PORT),
    "--dashboard-host", DASHBOARD_HOST,
    "--num-gpus", str(NUM_GPUS),
    "--block"   # keep Ray alive in this terminal
]

# Launch Ray head
print("Starting Ray head with command:", " ".join(cmd))
subprocess.run(cmd, check=True)
