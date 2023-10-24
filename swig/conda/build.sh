#!/bin/bash

build_threads=${FLAMEGPU_CONDA_BUILD_THREADS:-1}
[[ -z "$FLAMEGPU_CONDA_CUDA_ARCHITECTURES" ]] && build_arch="" || build_arch="-DCMAKE_CUDA_ARCHITECTURES=$FLAMEGPU_CONDA_CUDA_ARCHITECTURES"

# Setup python packages
# Issues with CMake locating ones installed by conda

pip install setuptools wheel build astpretty

mkdir -p build && cd build

#cmake .. -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_BUILD_PYTHON=ON -DFLAMEGPU_BUILD_PYTHON_CONDA=ON
#temp faster build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_BUILD_PYTHON=ON -DFLAMEGPU_BUILD_PYTHON_VENV=OFF -DFLAMEGPU_BUILD_PYTHON_CONDA=ON $build_arch

# Build python wheel
cmake --build . --target pyflamegpu --parallel ${build_threads}

# Install built wheel
pip install lib/Release/python/dist/pyflamegpu-2.0.0rc1+cuda122-cp38-cp38-linux_x86_64.whl

# Cleanup
cd .
rm -rf build