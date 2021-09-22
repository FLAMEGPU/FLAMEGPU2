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
set(THRUST_DOWNLOAD_VERSION 1.13.0)

# Use the FindCUDATooklit package (CMake > 3.17) to get the CUDA version and CUDA include directories for cub/thrust location hints
find_package(CUDAToolkit REQUIRED)

# Quietly find Thrust and CUB, to check if an appropriate version can be found without downloading.
find_package(Thrust QUIET CONFIG HINTS ${CUDAToolkit_INCLUDE_DIRS})
find_package(CUB QUIET CONFIG HINTS ${CUDAToolkit_INCLUDE_DIRS})

# If both were found with supported versions, there is no need to fetch Thrust via Fetch Content.
if(Thrust_FOUND AND Thrust_VERSION VERSION_GREATER_EQUAL MIN_REQUIRED_THRUST_VERSION AND CUB_FOUND AND CUB_VERSION VERSION_GREATER_EQUAL MIN_REQUIRED_CUB_VERSION)
    set(FETCH_THRUST_CUB 0)
    # Find the packages again but less quietly.
    find_package(Thrust CONFIG REQUIRED HINTS ${CUDAToolkit_INCLUDE_DIRS})
    find_package(CUB CONFIG REQUIRED HINTS ${CUDAToolkit_INCLUDE_DIRS})
else()
    # Otherwise set the variable indicating it needs to be downloaded, and re-set Cmake cache variables so it will be re-searched for.
    set(FETCH_THRUST_CUB 1)
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
        message(STATUS "Thrust/CUB >= ${MIN_REQUIRED_THRUST_VERSION} required, found ${Thrust_VERSION}. Downloading version ${THRUST_DOWNLOAD_VERSION}")

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
