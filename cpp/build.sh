#!/usr/bin/env bash
# Build fast_detector against the OpenVINO C++ runtime that ships inside the
# donate_env pip wheel (no system OpenVINO install needed).
set -euo pipefail

cd "$(dirname "$0")"
PROJECT_DIR="$(cd .. && pwd)"
OV="${PROJECT_DIR}/donate_env/lib/python3.12/site-packages/openvino"

if [[ ! -d "$OV/include" ]]; then
    echo "OpenVINO headers not found at $OV/include — activate/install donate_env first" >&2
    exit 1
fi

# The wheel ships libopenvino.so.2460 without the dev symlink, so link the
# versioned file directly and bake in an rpath to the wheel's libs dir.
LIBOV="$(ls "$OV"/libs/libopenvino.so.* | head -1)"

# manylinux OpenVINO wheels are built with the old libstdc++ string ABI
g++ -O3 -march=native -std=c++17 -Wall -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"$OV/include" \
    fast_detector.cpp \
    "$LIBOV" \
    -Wl,-rpath,"$OV/libs" \
    -lpthread \
    -o fast_detector

echo "Built: $(pwd)/fast_detector"
