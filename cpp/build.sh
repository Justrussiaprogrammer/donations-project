#!/usr/bin/env bash
# Build fast_detector via CMake against the OpenVINO C++ runtime shipped inside
# the donate_env pip wheel (no system OpenVINO install needed). Works on Linux
# and macOS. On Windows, run the equivalent cmake commands (see README).
#
# Extra cmake flags pass through, e.g.:
#   ./cpp/build.sh -DNATIVE_ARCH=OFF          # portable binary (no -march=native)
#   ./cpp/build.sh -DOpenVINO_DIR=/opt/...    # use a system OpenVINO instead
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found. Install it first:" >&2
    echo "  Ubuntu: sudo apt install -y cmake" >&2
    echo "  macOS:  brew install cmake" >&2
    exit 1
fi

if [[ ! -d "../donate_env" ]]; then
    echo "warning: ../donate_env not found — OpenVINO will be located via find_package." >&2
    echo "         Activate/create the venv and 'pip install -r requirements.txt' first," >&2
    echo "         or pass -DOpenVINO_DIR=... to point at a system OpenVINO." >&2
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build --config Release -j

echo "Built: $(pwd)/fast_detector"
