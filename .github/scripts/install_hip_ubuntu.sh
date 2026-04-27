#! /usr/bin/env bash
# Note: this is a very initial HIP installation script for CI. This will need expanding / improving as support for a wider range of HIPs is added.
# Todo: Full version of this as a script, including:
# Todo: Respect input env vars
# Todo: better subpackage management etc
# Todo: 7.x is different thatn 6.x etc
# Todo: Ubuntu version differences

# Define the hip packages to install, in a version agnostic way, targetting as few as required for CI to succeed
HIP_PACKAGES_IN=(
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
    [ "$1" = "$2" ] && return 1 || version_ge $1 $2
}
# returns 0 (true) if a <= b
function version_le() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$1" ]
}
# returns 0 (true) if a < b
function version_lt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_le $1 $2
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
# build a 3-part version string for package names
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
HIP_PACKAGES=""
for package in "${HIP_PACKAGES_IN[@]}"
do : 
    # Append the hip version to the pacakge name
    HIP_PACKAGES+=" ${package}${HIP_MAJOR_MINOR_PATCH}"
done
echo "HIP_PACKAGES ${HIP_PACKAGES}"

# enable command printing for CI debugging
set -x

# following ubuntu instructions from:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html

# install deps
$USE_SUDO apt-get install -y python3-setuptools python3-wheel

# Signing key
$USE_SUDO mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | $USE_SUDO tee /etc/apt/keyrings/rocm.gpg > /dev/null

# register packages
$USE_SUDO tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${HIP_VERSION_STR} ${UBUNTU_NAME} main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/${HIP_VERSION_STR}/ubuntu ${UBUNTU_NAME} main
EOF
$USE_SUDO tee /etc/apt/preferences.d/rocm-pin-600 << EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF
$USE_SUDO apt update

# Install packages.
$USE_SUDO apt install -y ${HIP_PACKAGES}

## -----------------
## Set environment vars / vars to be propagated
## -----------------

ROCM_PATH=/opt/rocm-${HIP_MAJOR_MINOR_PATCH}
echo "ROCM_PATH=${ROCM_PATH}"
export ROCM_PATH=${ROCM_PATH}
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"

# Check hipcc is now available.
hipcc -V

# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${ROCM_PATH}
    echo "Adding HIP ${HIP_MAJOR_MINOR_PATCH} to ROCM_PATH, PATH and LD_LIBRARY_PATH"
    echo "ROCM_PATH=${ROCM_PATH}" >> $GITHUB_ENV
    echo "${ROCM_PATH}/bin" >> $GITHUB_PATH
    echo "LD_LIBRARY_PATH=${ROCM_PATH}/bin/:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi
