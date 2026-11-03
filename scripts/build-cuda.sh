#!/bin/bash
# build-cuda.sh [dir]: ik_llama.cpp with r820.patch, CUDA build for the Quadro P2200 (Pascal, sm_61).
# Needs libsimde-dev (0.8.2); nvcc 12.4 needs gcc <= 13 as host compiler.
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
IK=${1:-$HOME/src/ik_llama.cpp}
[ -d "$IK" ] || git clone https://github.com/ikawrakow/ik_llama.cpp "$IK"
cd "$IK"
if ! git apply --reverse --check "$R"/ik_llama/r820.patch 2>/dev/null; then
    git checkout -q bdebfb6a0e8f92fc390e530a820bc245210d86db
    git apply "$R"/ik_llama/r820.patch
fi
cmake -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_IQK_AVX1_SIMDE=ON -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DLLAMA_CURL=OFF
cmake --build build-cuda -j "$(nproc)"
