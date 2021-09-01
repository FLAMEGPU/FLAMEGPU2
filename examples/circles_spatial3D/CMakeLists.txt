# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

# Name the project and set languages
project(circles_spatial3D CUDA CXX)

# Set the location of the ROOT flame gpu project relative to this CMakeList.txt
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. REALPATH)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Define output location of binary files
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # If top level project
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)
else()
    # If called via add_subdirectory()
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../bin/${CMAKE_BUILD_TYPE}/)
endif()

# Prepare list of source files
# Can't do this automatically, as CMake wouldn't know when to regen (as CMakeLists.txt would be unchanged)
SET(ALL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu
)

# Option to enable/disable building the static library
option(VISUALISATION "Enable visualisation support" OFF)

# Add the executable and set required flags for the target
add_flamegpu_executable("${PROJECT_NAME}" "${ALL_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)

# Also set as startup project (if top level project)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")

# Set the default (visual studio) debugger configure_file
set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                                                   VS_DEBUGGER_COMMAND_ARGUMENTS "-i 0.xml -s 200")