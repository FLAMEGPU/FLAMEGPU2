#! /usr/bin/env bash

set -e

# Define the hip packages to install, in a version agnostic way, targetting as few as required for CI to succeed

# Todo: rocm 7.14+ meta packages pull in binaries per gpu architecture, adding ~4 mins to download/install time, we don't need this in CI.
HIP_PACKAGES_IN_GE_7_14=(
    "amdrocm-core-dev"
)
HIP_PACKAGES_IN_LT_7_14=(
    "rocm-hip-runtime-dev"
    "rocthrust-dev"
    "hipcub-dev"
    "rocrand-dev"
    "hiprand-dev"
    "rocprofiler-sdk-roctx"
)

## -------------------
## Bash functions
## -------------------
# returns 0 (true) if a >= b
function version_ge() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$2" ]
}
# returns 0 (true) if a > b
function version_gt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_ge "$1" "$2"
}
# returns 0 (true) if a <= b
function version_le() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$1" ]
}
# returns 0 (true) if a < b
function version_lt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_le "$1" "$2"
}

## -----------------
## Check for root/sudo
## -----------------

# Detect if the script is being run as root, storing true/false in is_root.
is_root=false
if (( $EUID == 0)); then
   is_root=true
fi
# Find if sudo is available
has_sudo=false
if command -v sudo &> /dev/null ; then
    has_sudo=true
fi
# Decide if we can proceed or not (root or sudo is required) and if so store whether sudo should be used or not. 
if [ "$is_root" = false ] && [ "$has_sudo" = false ]; then 
    echo "Root or sudo is required. Aborting."
    exit 1
elif [ "$is_root" = false ] ; then
    USE_SUDO=sudo
else
    USE_SUDO=
fi


## ------------------
## Select HIP version
## ------------------

# Get the hip version from the environment as $hip.
HIP_VERSION_STR=${hip}

# Split the version.
# We (might/probably) don't know PATCH at this point - it depends which version gets installed.
HIP_MAJOR=$(echo "${HIP_VERSION_STR}" | cut -d. -f1)
HIP_MINOR=$(echo "${HIP_VERSION_STR}" | cut -d. -f2)
HIP_PATCH=$(echo "${HIP_VERSION_STR}" | cut -d. -f3)
# default patch to 0 if not provided
HIP_PATCH=${HIP_PATCH:-0}
# build a 2&3-part version string for package names
HIP_MAJOR_MINOR="${HIP_MAJOR}.${HIP_MINOR}"
HIP_MAJOR_MINOR_PATCH="${HIP_MAJOR}.${HIP_MINOR}.${HIP_PATCH}"
# use lsb_release to find the OS.
UBUNTU_NAME=$(lsb_release -sc)
UBUNTU_VERSION=$(lsb_release -sr)
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

echo "HIP_MAJOR: ${HIP_MAJOR}"
echo "HIP_MINOR: ${HIP_MINOR}"
echo "HIP_PATCH: ${HIP_PATCH}"
echo "UBUNTU_NAME: ${UBUNTU_NAME}"
echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"


# If we don't know the HIP_MAJOR or MINOR, error.
if [ -z "${HIP_MAJOR}" ] ; then
    echo "Error: Unknown HIP Major version. Aborting."
    exit 1
fi
if [ -z "${HIP_MINOR}" ] ; then
    echo "Error: Unknown HIP Minor version. Aborting."
    exit 1
fi
# If we don't know the Ubuntu version, error.
if [ -z ${UBUNTU_VERSION} ]; then
    echo "Error: Unknown Ubuntu version. Aborting."
    exit 1
fi

## ------------------------------
## Select HIP packages to install
## ------------------------------
# build  a space-separated list of packages to install via apt

if version_ge "$HIP_MAJOR_MINOR_PATCH" "7.14" ; then
    # >= 7.14 uses major.minor
    HIP_PACKAGES_IN=("${HIP_PACKAGES_IN_GE_7_14[@]}")
    HIP_PACKAGES=""
    for package in "${HIP_PACKAGES_IN[@]}"
    do : 
        # Append the hip version to the package name
        HIP_PACKAGES+=" ${package}${HIP_MAJOR_MINOR}"
    done
else
    # < 7.14 uses major.minor.path
    HIP_PACKAGES_IN=("${HIP_PACKAGES_IN_LT_7_14[@]}")
    HIP_PACKAGES=""
    for package in "${HIP_PACKAGES_IN[@]}"
    do : 
        # Append the hip version to the package name
        HIP_PACKAGES+=" ${package}${HIP_MAJOR_MINOR_PATCH}"
    done
fi

echo "HIP_PACKAGES ${HIP_PACKAGES}"

# enable command printing for CI debugging
set -x

# following ubuntu instructions from:
# https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman#rocm-install-meta-packages
# https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman#installation
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html

# install deps
$USE_SUDO apt-get install -y python3-setuptools python3-wheel libatomic1 libquadmath0

# Signing key and register the apt repository, depending on the rocm and ubuntu versions requested as this changes often with rocm...
# not indented due to EOF usage
if version_ge "$HIP_MAJOR_MINOR_PATCH" "10.0" ; then

# rocm 10.x
$USE_SUDO mkdir --parents --mode=0755 /etc/apt/keyrings
$USE_SUDO wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | gpg --dearmor | $USE_SUDO tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
$USE_SUDO tee /etc/apt/sources.list.d/amdrocm-stable.sources << EOF
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu${UBUNTU_VERSION}/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF
$USE_SUDO apt-get update

elif version_ge "$HIP_MAJOR_MINOR_PATCH" "7.14" ; then

# rocm 7.14
$USE_SUDO mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - | gpg --dearmor | $USE_SUDO tee /etc/apt/keyrings/amdrocm.gpg > /dev/null
$USE_SUDO tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu${UBUNTU_VERSION} stable main
EOF
$USE_SUDO apt-get update

else

# rocm <= 7.2 (down to 6.something?)
$USE_SUDO mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | $USE_SUDO tee /etc/apt/keyrings/rocm.gpg > /dev/null
$USE_SUDO tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${HIP_VERSION_STR} ${UBUNTU_NAME} main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/${HIP_VERSION_STR}/ubuntu ${UBUNTU_NAME} main
EOF
$USE_SUDO tee /etc/apt/preferences.d/rocm-pin-600 << EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF
$USE_SUDO apt-get update

fi

# Install packages.
$USE_SUDO apt-get install -y ${HIP_PACKAGES}

## -----------------
## Set environment vars / vars to be propagated
## -----------------

if version_ge "$HIP_MAJOR_MINOR_PATCH" "7.14" ; then
    ROCM_PATH=/opt/rocm/
else
   ROCM_PATH=/opt/rocm-${HIP_MAJOR_MINOR_PATCH}/
fi

echo "ROCM_PATH=${ROCM_PATH}"
export ROCM_PATH=${ROCM_PATH}
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${ROCM_PATH}
    echo "Adding HIP ${HIP_MAJOR_MINOR_PATCH} to ROCM_PATH, PATH and LD_LIBRARY_PATH"
    echo "ROCM_PATH=${ROCM_PATH}" >> $GITHUB_ENV
    echo "${ROCM_PATH}/bin" >> $GITHUB_PATH
    echo "LD_LIBRARY_PATH=${ROCM_PATH}/lib/:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi
