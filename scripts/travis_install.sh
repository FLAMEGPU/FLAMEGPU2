#!/bin/bash
# install dependencies
# (this script must be run as root)

# Usfull for debugging
set -ev

# Update
apt-get -y update

# Install Boost
apt-get install libboost-dev

# Install CUDA (see caffe example https://github.com/BVLC/caffe/blob/master/scripts/travis/install-deps.sh)
CUDA_REPO_PKG=cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
#CUDA_REPO_PKG=cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
#wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/$CUDA_REPO_PKG
dpkg -i $CUDA_REPO_PKG
rm $CUDA_REPO_PKG

# update package lists
apt-get -y update

# install packages
CUDA_PKG_VERSION="8-0"
CUDA_VERSION="8.0"
apt-get install -y --no-install-recommends \
  cuda-core-$CUDA_PKG_VERSION \
  cuda-cudart-dev-$CUDA_PKG_VERSION
#apt-get --yes --force-yes install cuda
  
# manually create CUDA symlink
ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
