#!/bin/bash

# Setup user config
build_threads=${FLAMEGPU_CONDA_BUILD_THREADS:-1}
[[ -z "$FLAMEGPU_CONDA_CUDA_ARCHITECTURES" ]] && build_arch="" || build_arch="-DCMAKE_CUDA_ARCHITECTURES=$FLAMEGPU_CONDA_CUDA_ARCHITECTURES"

# Locate SWIG
# (CMake can't auto find conda installed SWIG)
swig_exe=$(find ${CONDA_PREFIX} -type f -name swig -print -quit)
swig_dir=$(dirname $(find ${CONDA_PREFIX} -type f -name swig.swg -print -quit))
if [[ ! -z "$swig_exe" ]] && [[ ! -z "$swig_dir" ]]; then
swig_exe="-DSWIG_EXECUTABLE=$swig_exe"
swig_dir="-DSWIG_DIR=$swig_dir"
else
swig_exe=""
swig_dir=""
fi
echo "swig_exe: $swig_exe"
echo "swig_dir: $swig_dir"
# Setup python packages
# Issues with CMake locating ones installed by conda

#pip install setuptools wheel build astpretty

mkdir -p build && cd build

# Configure CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_BUILD_PYTHON=ON -DFLAMEGPU_BUILD_PYTHON_VENV=OFF -DFLAMEGPU_BUILD_PYTHON_CONDA=ON $build_arch $swig_exe $swig_dir

# Build Python wheel
#cmake --build . --target pyflamegpu --parallel ${build_threads}

# Install built wheel
#pip install lib/Release/python/dist/pyflamegpu-2.0.0rc1+cuda122-cp38-cp38-linux_x86_64.whl

# Cleanup
cd .
rm -rf build