##############################
# Thrust (and CUB) >= 1.9.10 #
##############################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Thrust includes CUB as a git submodule, at ${thrust_SOURCE_DIR}/dependencies/cub, with a symlink pointing to it from ${thrust_SOURCE_DIR/cub}. 
# Under windows, git by default cannot create symlinks (it can be enabled when installing if the user has sufficient priviledges, but this cannot be relied upon)
# Instead, we check if CUB is accessible via the symlink, otherwise we check the expected dependency location.
# This may need some adjusting for future Thrust versions (potentially)

# Declare information about where and what we want from thrust.
# Thrust version must be >= 1.9.10 for good cmake integration, making our lives easy
# We require 1.9.8 for a bug fix anyway, so no harm using 1.9.10 instead.
FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    GIT_TAG        1.9.10
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
    # Check that CUB exists at the expected (symlinked) location.
    if(EXISTS "${EXPECTED_CUB_CONFIG_LOCATION}")
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
