#!/bin/bash
# install dependencies
# (this script must be run as root)

# Update
apt-get -y update

# Install CUDA (see caffe example https://github.com/BVLC/caffe/blob/master/scripts/travis/install-deps.sh)
CUDA_REPO_PKG=cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
rm $CUDA_REPO_PKG

# update package lists
apt-get -y update

# install packages
CUDA_PKG_VERSION="8-0"
CUDA_VERSION="8.0"
apt-get install -y --no-install-recommends \
  cuda-core-$CUDA_PKG_VERSION
  
# manually create CUDA symlink
# ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

# add nvcc to PATH
export PATH="$PATH:/usr/local/cuda-$CUDA_VERSION/bin"
