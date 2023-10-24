#!/bin/bash -v

# Enable output of bash commands executed to make script debugging easier.
set -x 

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
cmake .. -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_BUILD_PYTHON=ON -DFLAMEGPU_BUILD_PYTHON_VENV=OFF -DFLAMEGPU_BUILD_PYTHON_CONDA=ON $build_arch $swig_exe $swig_dir -DPython_EXECUTABLE="$PYTHON" $CMAKE_ARGS

# Build Python wheel
cmake --build . --target pyflamegpu --parallel $build_threads

# Install built wheel
pyfgpu_wheel=$(find "lib/Release/python/dist/" -type f -name "pyflamegpu*.whl" -print -quit)
pip install --no-deps $pyfgpu_wheel

# Cleanup
cd ..
rm -rf build