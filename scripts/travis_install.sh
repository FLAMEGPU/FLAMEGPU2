#!/bin/bash
# install dependencies
# (this script must be run as root)

# Usfull for debugging
set -ev

# Get some variables
UBUNTU_CODENAME=$(lsb_release -c | cut -f2)

# Update
apt-get -y update

# Install apt dependencies
apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget libboost-dev python3-pip doxygen

# Install recent CMAKE via kitware APT repo
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main"
sudo apt update -qq 
sudo apt install -y cmake


# Install CUDA (see https://github.com/jeremad/cuda-travis/blob/master/.travis.yml)
CUDA_REPO_PKG=cuda-repo-${UBUNTU_VERSION}_${CUDA_LONG}_amd64.deb
CUDA_PACKAGE_VERSION=${CUDA_SHORT/./-}

# Download and install CUDA repo installer
wget http://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
rm ${CUDA_REPO_PKG}

# Add nvidia public key
wget https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/7fa2af80.pub
apt-key add 7fa2af80.pub

# Update package lists
apt-get -qq update

# Install CUDA packages
apt-get install -y --no-install-recommends cuda-core-${CUDA_PACKAGE_VERSION} cuda-cudart-dev-${CUDA_PACKAGE_VERSION}

# Install cpplint (optional, not currently used at CI time)
# pip3 install cpplint
