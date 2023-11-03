#!/bin/bash


[[ -z "$FLAMEGPU_CONDA_CUDA_ARCHITECTURES" ]] && build_arch="" || build_arch="-DCMAKE_CUDA_ARCHITECTURES=$FLAMEGPU_CONDA_CUDA_ARCHITECTURES"

echo "${build_arch}"
echo ${build_arch}
echo $build_arch