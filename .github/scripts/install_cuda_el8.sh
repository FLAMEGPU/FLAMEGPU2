# Install CUDA on Alma8/manylinux_2_28.

## -------------------
## Constants
## -------------------

# dnf install cuda-nvrtc-devel-11-4 cuda-compiler-11-4 cuda-cudart-devel-11-4 cuda-nvcc-11-4 cuda-nvrtc-11-4 cuda-nvtx-11-4 libcurand-devel-11-4

# List of sub-packages to install.
# @todo - pass this in from outside the script?
# @todo - check the specified subpackages exist via apt pre-install?  apt-rdepends cuda-9-0 | grep "^cuda-"?

# Ideally choose from the list of meta-packages to minimise variance between cuda versions (although it does change too)
CUDA_PACKAGES_IN=(
    "cuda-compiler"
    "cuda-cudart-devel" # libcudart.so
    "cuda-driver-devel" # libcuda.so
    "cuda-nvtx"
    "cuda-nvrtc-devel"
    "libcurand-devel" # 11-0+
    "libnvjitlink-devel" #12-0+
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


## -------------------
## Select CUDA version
## -------------------

# Get the cuda version from the environment as $cuda.
CUDA_VERSION_MAJOR_MINOR=${cuda}

# Split the version.
# We (might/probably) don't know PATCH at this point - it depends which version gets installed.
CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)
# query rpm to find the major enterprise linux release
EL_MAJOR=$(rpm -E %{rhel})

echo "CUDA_MAJOR: ${CUDA_MAJOR}"
echo "CUDA_MINOR: ${CUDA_MINOR}"
echo "CUDA_PATCH: ${CUDA_PATCH}"
echo "EL_MAJOR:   ${EL_MAJOR}"

# If we don't know the CUDA_MAJOR or MINOR, error.
if [ -z "${CUDA_MAJOR}" ] ; then
    echo "Error: Unknown CUDA Major version. Aborting."
    exit 1
fi
if [ -z "${CUDA_MINOR}" ] ; then
    echo "Error: Unknown CUDA Minor version. Aborting."
    exit 1
fi
# If we don't know the Ubuntu version, error.
if [ -z ${EL_MAJOR} ]; then
    echo "Error: Unknown EL version. Aborting."
    exit 1
fi

## -------------------------------
## Select CUDA packages to install
## -------------------------------
CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"
do :
    # CUDA < 11, lib* packages were actually cuda-cu* (generally, this might be greedy.)
    if [[ ${package} == libcu* ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "11.0" ; then
        package="${package/libcu/cuda-cu}"
    fi
    # libnvjitlink not required prior to CUDA 12.0
    if [[ ${package} == libnvjitlink-dev* ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "12.0" ;then
        continue
    fi
    # CUDA < 11, -devel- packages were actually -dev
    if [[ ${package} == *devel* ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "11.0" ; then
        package="${package//devel/dev}"
    fi
    # Build the full package name and append to the string.
    CUDA_PACKAGES+=" ${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
done
echo "CUDA_PACKAGES ${CUDA_PACKAGES}"

## -----------------
## Prepare to install
## -----------------

CPU_ARCH="x86_64"
# Nvidia don't provide an explicit alma repo. 11.2's closest is RHEL.
# 12.4 includes rocky8/9, rhel7/8/9, cent7, so RHEL is the closes that should hopefully be fine, otherwise will have to switch to the much slower runfile installer.
DNF_REPO_URI="https://developer.download.nvidia.com/compute/cuda/repos/rhel${EL_MAJOR}/${CPU_ARCH}/cuda-rhel${EL_MAJOR}.repo"

echo "DNF_REPO_URI ${DNF_REPO_URI}"

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

## -----------------
## Install
## -----------------
echo "Adding CUDA Repository"
$USE_SUDO dnf config-manager --add-repo ${DNF_REPO_URI}
$USE_SUDO dnf clean all

echo "Installing CUDA packages ${CUDA_PACKAGES}"
$USE_SUDO dnf -y install ${CUDA_PACKAGES}

if [[ $? -ne 0 ]]; then
    echo "CUDA Installation Error."
    exit 1
fi

## -----------------
## Set environment vars / vars to be propagated
## -----------------

CUDA_PATH=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
echo "CUDA_PATH=${CUDA_PATH}"
export CUDA_PATH=${CUDA_PATH}
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
# Check nvcc is now available.
nvcc -V

# If executed on github actions, make the appropriate echo statements to update the environment
if [[ $GITHUB_ACTIONS ]]; then
    # Set paths for subsequent steps, using ${CUDA_PATH}
    echo "Adding CUDA to CUDA_PATH, PATH and LD_LIBRARY_PATH"
    echo "CUDA_PATH=${CUDA_PATH}" >> $GITHUB_ENV
    echo "${CUDA_PATH}/bin" >> $GITHUB_PATH
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
    echo "LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}" >> $GITHUB_ENV
fi