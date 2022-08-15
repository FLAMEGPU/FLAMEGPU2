###################################
# Flamegpu2 Visualisation Library #
###################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Set the visualiser repo and tag to use unless overridden by the user.
# @todo - If the git version has changed in this file, fetch again? 
set(DEFAULT_VISUALISATION_GIT_VERSION "ef676da1daf8d8118bc7b2b75ede958c4f08afc9")
set(DEFAULT_VISUALISATION_REPOSITORY "https://github.com/FLAMEGPU/FLAMEGPU2-visualiser.git")

# If overridden by the user, attempt to use that
if (VISUALISATION_ROOT)
    # @todo - we should make the visualisation package find_package() compatible, and check it exists if VISUALISATION_ROOT is set. 

    # Look for the main visualisation header to get the abs path, but only look relative to the hints/paths, no cmake defaults (for now)
    set(VISUALISATION_INCLUDE_HEADER_FILE include/flamegpu/visualiser/FLAMEGPU_Visualisation.h)
    find_path(VISUALISATION_ROOT_ABS
        NAMES
            ${VISUALISATION_INCLUDE_HEADER_FILE}
        HINTS
            ${VISUALISATION_ROOT}
        PATHS
            ${VISUALISATION_ROOT}
        NO_DEFAULT_PATH
    )
    # If found, use the local vis, otherwise error.
    if(VISUALISATION_ROOT_ABS)
        # If the correct visualtiion root was found, output a successful status message
        message(STATUS "Found VISUALISATION_ROOT: ${VISUALISATION_ROOT_ABS} (${VISUALISATION_ROOT})")
        # update the value to the non abs version, in local and parent scope.
        set(VISUALISATION_ROOT "${VISUALISATION_ROOT_ABS}")
        set(VISUALISATION_ROOT "${VISUALISATION_ROOT_ABS}" PARENT_SCOPE)
        # And set up the visualisation build 
        add_subdirectory(${VISUALISATION_ROOT_ABS} ${CMAKE_CURRENT_BINARY_DIR}/_deps/flamegpu_visualiser-build EXCLUDE_FROM_ALL)
        # Set locally and for parent scope, which are mutually exclusive
        set(VISUALISATION_BUILD ${flamegpu_visualiser_BINARY_DIR} CACHE INTERNAL "flamegpu_visualiser_BINARY_DIR")
    else()
        # Send a fatal error if the visualstion root passed is invalid.
        message(FATAL_ERROR "Invalid VISUALISATION_ROOT '${VISUALISATION_ROOT}'.\nVISUALISATION_ROOT must be a valid directory containing '${VISUALISATION_INCLUDE_HEADER_FILE}'")
    endif()

else()
    # If a VISUALISATION_GIT_VERSION has not been defined, set it to the default option.
    if(NOT DEFINED VISUALISATION_GIT_VERSION OR VISUALISATION_GIT_VERSION STREQUAL "")
        set(VISUALISATION_GIT_VERSION "${DEFAULT_VISUALISATION_GIT_VERSION}" CACHE STRING "Git branch or tag to use for the FLAMEPGU2_visualiaer")
    endif()

    # Allow users to switch to forks with relative ease.
    if(NOT DEFINED VISUALISATION_REPOSITORY OR VISUALISATION_REPOSITORY STREQUAL "")
        set(VISUALISATION_REPOSITORY "${DEFAULT_VISUALISATION_REPOSITORY}" CACHE STRING "Remote Git Repository for the FLAMEPGU2_visualiaer")
    endif()

    # Otherwise download.
    FetchContent_Declare(
        flamegpu_visualiser
        GIT_REPOSITORY ${VISUALISATION_REPOSITORY}
        GIT_TAG        ${VISUALISATION_GIT_VERSION}
        GIT_PROGRESS   ON
        # UPDATE_DISCONNECTED   ON
        )
        FetchContent_GetProperties(flamegpu_visualiser)
    if(NOT flamegpu_visualiser_POPULATED)
        message(STATUS "using flamegpu_visualiser ${VISUALISATION_GIT_VERSION} from ${VISUALISATION_REPOSITORY}")
        FetchContent_Populate(flamegpu_visualiser)
    
        add_subdirectory(${flamegpu_visualiser_SOURCE_DIR} ${flamegpu_visualiser_BINARY_DIR} EXCLUDE_FROM_ALL)
        
        # Set locally and for parent scope, which are mutually exclusive
        set(VISUALISATION_ROOT ${flamegpu_visualiser_SOURCE_DIR} CACHE INTERNAL "flamegpu_visualiser_SOURCE_DIR")
        set(VISUALISATION_BUILD ${flamegpu_visualiser_BINARY_DIR} CACHE INTERNAL "flamegpu_visualiser_BINARY_DIR")
    endif()
endif()
unset(DEFAULT_VISUALISATION_GIT_VERSION)
unset(DEFAULT_VISUALISATION_REPOSITORY)
