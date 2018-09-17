# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, 3.12 is better to fix some device linker errors
cmake_minimum_required(VERSION VERSION 3.10 FATAL_ERROR)

# Tests require boost
find_package(Boost 1.54 REQUIRED)

# Name the project and set languages
project(tests CUDA CXX)

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. REALPATH)

# Define output location of binary files
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${FLAMEGPU_ROOT}/bin/linux-x64/${CMAKE_BUILD_TYPE}/) # Global dir
# SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}/) # Project specific

# Prepare list of source files
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_all.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/device_functions.cu
)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Add the executable and set required flags for the target
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}")
