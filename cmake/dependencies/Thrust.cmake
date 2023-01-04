####################
# Thrust (and CUB) #
####################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Set the minimum supported cub/thrust version, and the version to fetch
# Thrust minimum version to 1.16 to avoid windows.h related issues and pull in bug fixes, but fetch the most recent 1.x release otherwise (at the time of writing).
set(MIN_REQUIRED_THRUST_VERSION 1.16.0)
set(MIN_REQUIRED_CUB_VERSION ${MIN_REQUIRED_THRUST_VERSION})
set(THRUST_DOWNLOAD_VERSION 1.17.2)

# Use the FindCUDATooklit package (CMake > 3.17) to get the CUDA version and CUDA include directories for cub/thrust location hints
find_package(CUDAToolkit REQUIRED)

# Quietly find Thrust and CUB, to check if an appropriate version can be found without downloading.
# thrust-config.cmake and cub-config.cmake live in different locations with CUDA (on ubuntu) depending on the CUDA version.
# CUDA 11.3 and 11.4 they can be found in the CUDA Toolkit include directories.
# CUDA 11.5+ they can be found in lib/cmake or lib64/cmake
# CUDA 11.6 - 11.8 ships with CUB 1.15.0 which has a bug when windows.h is included prior to CUB, so don't try to find the regular Thrust/CUB in this case. 
# Ideally we would detect 1.15.0 and then download the correct version of CUB/Thrust, but getting CMake on windows to behave was proving problematic
if(NOT (MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.6.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.9.0))
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
    # Unset a number of thrust / cub cache variables so that re-finding behaves as intended.
    unset(THRUST_DIR)
    unset(THRUST_DIR CACHE)
    unset(THRUST_DEVICE_SYSTEM_OPTIONS)
    unset(THRUST_DEVICE_SYSTEM_OPTIONS CACHE)
    unset(THRUST_HOST_SYSTEM_OPTIONS)
    unset(THRUST_HOST_SYSTEM_OPTIONS CACHE)
    unset(THRUST_VERSION)
    unset(THRUST_VERSION CACHE)
    unset(THRUST_VERSION_COUNT)
    unset(THRUST_VERSION_COUNT CACHE)
    unset(THRUST_VERSION_MAJOR)
    unset(THRUST_VERSION_MAJOR CACHE)
    unset(THRUST_VERSION_MINOR)
    unset(THRUST_VERSION_MINOR CACHE)
    unset(THRUST_VERSION_PATCH)
    unset(THRUST_VERSION_PATCH CACHE)
    unset(THRUST_VERSION_TWEAK)
    unset(THRUST_VERSION_TWEAK CACHE)
    unset(_THRUST_CMAKE_DIR)
    unset(_THRUST_CMAKE_DIR CACHE)
    unset(_THRUST_INCLUDE_DIR)
    unset(_THRUST_INCLUDE_DIR CACHE) # This is the most important one for Thrust 2.0, which just THRUST_DIR was insufficient for.
    unset(_THRUST_QUIET)
    unset(_THRUST_QUIET CACHE)
    unset(_THRUST_QUIET_FLAG)
    unset(_THRUST_QUIET_FLAG CACHE)
    unset(CUB_DIR)
    unset(CUB_DIR CACHE)
    unset(_CUB_INCLUDE_DIR)
    unset(_CUB_INCLUDE_DIR CACHE)
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
        message(STATUS "Fetching Thrust ${THRUST_DOWNLOAD_VERSION}")
        FetchContent_Populate(thrust)
        # Use find_package for thrust, only looking for the fetched version.
        # This creates a non-system target due to nvcc magic to avoid the cuda toolkit version being used instead, so warnings are not suppressable.
        find_package(Thrust REQUIRED CONFIG
            PATHS ${thrust_SOURCE_DIR}
            NO_CMAKE_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_PACKAGE_REGISTRY
            NO_CMAKE_SYSTEM_PATH)
        # Use find_package for cub, only looking for the fetched version.
        # This creates a non-system target due to nvcc magic to avoid the cuda toolkit version being used instead, so warnings are not suppressable.
        # Look in the symlinked and non-symlinked locations, preferring non symlinked due to windows (and the symlink being removed from 2.0)
        find_package(CUB REQUIRED CONFIG
            PATHS
                ${thrust_SOURCE_DIR}/dependencies/cub/cub/cmake/
                ${thrust_SOURCE_DIR}/cub/cmake
            NO_CMAKE_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_PACKAGE_REGISTRY
            NO_CMAKE_SYSTEM_PATH)
    endif()
    # Mark some CACHE vars as advnaced for a cleaner CMake GUI
    mark_as_advanced(FETCHCONTENT_QUIET)
    mark_as_advanced(FETCHCONTENT_BASE_DIR)
    mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_THRUST)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_THRUST)
endif()

# Unset temporary variables
unset(FETCH_THRUST_CUB)
unset(MIN_REQUIRED_THRUST_VERSION)
unset(MIN_REQUIRED_CUB_VERSION)
unset(THRUST_DOWNLOAD_VERSION)

# Mark some CACHE vars as advnaced for a cleaner CMake GUI
mark_as_advanced(CUB_DIR)
mark_as_advanced(Thrust_DIR)