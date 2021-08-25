#! /usr/bin/env bash

# Script to use auditwheel repair to package .so files into the python wheel, excluding cuda libraries
# This keeps wheel size small, but does not produce manylinux compliant wheels. dlopen must be used instead, in which case regurally running auditwheel repair would be viable.

# This will not be required once libcuda.so etc ar dlopen'd instead, and regular auditwheel repair can be used.

# Takes the path to a wheel to fixup as the first argument.

SO_TO_NOT_PACKAGE=(
    libcuda.so
    libcudart.so
    libnvrtc.so
    libcurand.so
    libGLdispatch.so
    libGLU.so
    libGLX.so
    libOpenGL.so
)


ARGC_REQUIRED=2
function print_usage {
    scriptname=$(basename $0)
    echo "Usage:"
    echo "    ${scriptname} <path/to/file.whl> <platform>"
    echo ""
    echo "    <path/to/file.whl>: Path to the wheelfile to fixup."
    echo "    <platform>: The platform to target, i.e. manylinux2014_x86_64"
}


if [ $# -ne ${ARGC_REQUIRED} ]; then
    scriptname=$(basename $0)
    echo "Error: ${scriptname} requires ${ARGC_REQUIRED} arguments"
    print_usage
    exit 1
fi
WHEEL_PATH=$1
PLATFORM=$2
echo "Fixing up ${WHEEL_PATH}"
echo "Using auditwheel platform ${PLATFORM}"


# Check for requirements, patchelf and auditwheel, unzip and zip
AUDITWHEEL=$(command -v auditwheel)
if [ $? -ne 0 ]; then
    echo "Error: auditwheel is required."
    exit 1
fi
PATCHELF=$(command -v patchelf)
if [ $? -ne 0 ]; then
    echo "Error: patchelf is required."
    exit 1
fi
ZIP=$(command -v zip)
if [ $? -ne 0 ]; then
    echo "Error: zip is required."
    exit 1
fi
UNZIP=$(command -v unzip)
if [ $? -ne 0 ]; then
    echo "Error: unzip is required."
    exit 1
fi

# Check that the input wheel exit 1s.
if [ ! -f "${WHEEL_PATH}" ]; then
    echo "Error: input wheel ${WHEEL_PATH} does not exist"
    exit 1
fi

# grep the output of manywheel repair --help to check for the platform. Error if its missing.
platform_match=$(auditwheel repair --help | grep "${PLATFORM}")
if [ $? -ne 0 ]; then
    echo "Error: provided platform ${PLATFORM} does not appear to be supported by auditwheel."
    auditwheel repair --help | grep "\- " | cut -d"-" -f2
    exit 1
fi

WHEEL_NAME=$(basename ${WHEEL_PATH})
WHEEL_DIR=$(dirname ${WHEEL_PATH})
EXTRACTED_DIRNAME="extracted"
EXTRACTED_DIR="${WHEEL_DIR}/${EXTRACTED_DIRNAME}"

echo "SO_TO_NOT_PACKAGE ${SO_TO_NOT_PACKAGE}"
echo "WHEEL_NAME ${WHEEL_NAME}"
echo "WHEEL_PATH ${WHEEL_PATH}"
echo "WHEEL_DIR ${WHEEL_DIR}"
echo "EXTRACTED_DIRNAME ${EXTRACTED_DIRNAME}"
echo "EXTRACTED_DIR ${EXTRACTED_DIR}"

# Variable to contain SO's that were found and must be re-packaged.
SO_TO_REPACKAGE=()

set -x

# Ensure the extraction directory exists
mkdir -p "${EXTRACTED_DIR}"

###############
# Extract the wheel
###############
unzip -o "${WHEEL_PATH}" -d "${EXTRACTED_DIR}"

# Switch into the extracted directory
pushd ${EXTRACTED_DIR}

# Find the .so file to modify
SOFILE=$(find . -name "_*.so" | head -n 1)
echo "SOFILE=${SOFILE}" 

# Output the needed .so's for debugging info
patchelf --print-needed ${SOFILE}

# Grab the output of ldd once. 
LDD_OUTPUT=$(ldd "${SOFILE}")

# Remove matching .so's via patchelf
for SOPATTERN in "${SO_TO_NOT_PACKAGE[@]}"; do
    S=$(echo "${LDD_OUTPUT}" | grep ${SOPATTERN} | sed -e 's/^[[:space:]]*//' | cut -d " " -f 1 | head -n 1)
    if [ ! -z "$S" ] ; then
        echo "removing ${S}"
        patchelf --remove-needed ${S} ${SOFILE}
        SO_TO_REPACKAGE+=($S)
    fi
done

# Output the needed .so's for debugging info
patchelf --print-needed ${SOFILE}

# Repackage the wheel.
zip -r ${WHEEL_NAME} *

# Pop out of the extracted dir
popd

# Copy the wheel back to it's original location.
mv ${EXTRACTED_DIR}/${WHEEL_NAME} ${WHEEL_PATH}

# Clean the extraction directory
rm -rf ${EXTRACTED_DIR}/*


###############
# Run auditwheel repair
###############

# Use --no-update-tags to not rename the file
# Use --only-plat to not suggest newer higher compatability versions
auditwheel repair --plat ${PLATFORM} --no-update-tags --only-plat -w ${WHEEL_DIR} ${WHEEL_PATH}

# Check if  auditwheel repair was successful or not. 
if [ $? -ne 0 ]; then
    echo "Error: Auditwheel error. Cannot proceed."
    exit 1
fi


###############
# Re-add .so's
###############

# Extract the repaird wheel
unzip -o "${WHEEL_PATH}" -d "${EXTRACTED_DIR}"

# Switch into the extracted directory
pushd "${EXTRACTED_DIR}"

# Find the .so file to modify
SOFILE=$(find . -name "_*.so" | head -n 1)
echo "SOFILE=${SOFILE}" 

# Output the needed .so's for debugging info
patchelf --print-needed ${SOFILE}

# Add removed .so's via patchelf
for S in "${SO_TO_REPACKAGE[@]}"; do
    if [ ! -z "$S" ] ; then
        echo "re-adding ${S}"
        patchelf --add-needed ${S} ${SOFILE}
    fi
done

# Output the needed .so's for debugging info
patchelf --print-needed ${SOFILE}

# Repackage the wheel.
zip -r ${WHEEL_NAME} *

# Pop out of the extracted dir
popd

# Copy the wheel back to it's original location.
mv "${EXTRACTED_DIR}/${WHEEL_NAME}" "${WHEEL_PATH}"

# Clean the extraction directory
rm -rf ${EXTRACTED_DIR}/*

# Delete the extraction directory
rm -rf ${EXTRACTED_DIR}