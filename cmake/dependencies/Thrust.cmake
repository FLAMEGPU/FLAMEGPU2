####################
# Thrust (and CUB) #
####################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Set the minimum supported cub/thrust version, and the version to fetch
# Thrust version must be >= 1.9.10 for good cmake integration. 
# Thrust 1.9.8 is a miniumum as it includes a bugfix that was causing issues.
set(MIN_REQUIRED_THRUST_VERSION 1.11.0)
set(MIN_REQUIRED_CUB_VERSION ${MIN_REQUIRED_THRUST_VERSION})
set(THRUST_DOWNLOAD_VERSION 1.14.0)

# Use the FindCUDATooklit package (CMake > 3.17) to get the CUDA version and CUDA include directories for cub/thrust location hints
find_package(CUDAToolkit REQUIRED)

# Quietly find Thrust and CUB, to check if an appropriate version can be found without downloading.
# thrust-config.cmake and cub-config.cmake live in different locations with CUDA (on ubuntu) depending on the CUDA version.
# CUDA 11.3 and 11.4 they can be found in the CUDA Toolkit include directories.
# CUDA 11.5+ they can be found in lib/cmake or lib64/cmake
# CUDA 11.6 (and 11.7) ships with CUB 1.15.0 which has a bug when windows.h is included prior to CUB, so don't try to find the regular Thrust/CUB in this case. 
# Ideally we would detect 1.15.0 and then download the correct version of CUB/Thrust, but getting CMake on windows to behave was proving problematic
if(NOT (MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.6.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8.0))
    find_package(Thrust QUIET CONFIG HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_DIR}/cmake)
    find_package(CUB QUIET CONFIG HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_DIR}/cmake)
endif()

# By default, assume we have to fetch thrust/cub
set(FETCH_THRUST_CUB 1)
# If a useful version was found, find it again less quietly 
if(Thrust_FOUND AND Thrust_VERSION VERSION_GREATER_EQUAL MIN_REQUIRED_THRUST_VERSION AND CUB_FOUND AND CUB_VERSION VERSION_GREATER_EQUAL MIN_REQUIRED_CUB_VERSION)
    set(FETCH_THRUST_CUB 0)
    # Find the packages again but less quietly.
    find_package(Thrust CONFIG REQUIRED HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_DIR}/cmake)
    find_package(CUB CONFIG REQUIRED HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_DIR}/cmake)
# Otherwise unfind Thrust/CUB.
else()
    # As CONFIG mode was used, only <PackageName>_DIR should need deleting for latter calls to find_package to work.
    unset(Thrust_DIR CACHE)
    unset(Thrust_DIR)
    unset(CUB_DIR CACHE)
    unset(CUB_DIR)
endif()

# If thrust/cub do need downloading, fetch them, and find them.
# As they are header only, they can just be found rather than add_subdirectoried.
if(FETCH_THRUST_CUB)
    # Declare information about where and what we want from thrust.
    FetchContent_Declare(
        thrust
        GIT_REPOSITORY https://github.com/NVIDIA/thrust.git
        GIT_TAG        ${THRUST_DOWNLOAD_VERSION}
        GIT_SHALLOW    1
        GIT_PROGRESS   ON
        # UPDATE_DISCONNECTED   ON
    )

    # Fetch and populate the content if required.
    FetchContent_GetProperties(thrust)
    if(NOT thrust_POPULATED)
        FetchContent_Populate(thrust)
        # Add thrusts' expected location to the prefix path.
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${thrust_SOURCE_DIR}/thrust/cmake")
        # Set the location for where to find cub (ideally)
        set(EXPECTED_CUB_CONFIG_LOCATION "${thrust_SOURCE_DIR}/cub/cmake/")
        # Thrust includes CUB as a git submodule, at ${thrust_SOURCE_DIR}/dependencies/cub, with a symlink pointing to it from ${thrust_SOURCE_DIR/cub}. 
        # Under windows, git by default cannot create symlinks (it can be enabled when installing if the user has sufficient priviledges, but this cannot be relied upon)
        # Instead, we check if CUB is accessible via the symlink, otherwise we check the expected dependency location.
        # This may need some adjusting for future Thrust versions (potentially)
        if(EXISTS "${EXPECTED_CUB_CONFIG_LOCATION}" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # Use the symlinked "default" location
            set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${thrust_SOURCE_DIR}/cub/cmake")
        else()
            # Otherwise, use the non-symlinked location.
            set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${thrust_SOURCE_DIR}/dependencies/cub/cub/cmake/")
        endif()
        # Use find_package for thrust and cub, which are required.
        find_package(Thrust REQUIRED CONFIG)
        find_package(CUB REQUIRED CONFIG)
    endif()
endif()

# Unset temporary variables
unset(FETCH_THRUST_CUB)
unset(MIN_REQUIRED_THRUST_VERSION)
unset(MIN_REQUIRED_CUB_VERSION)
unset(THRUST_DOWNLOAD_VERSION)
