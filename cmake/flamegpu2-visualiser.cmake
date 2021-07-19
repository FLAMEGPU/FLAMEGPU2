###################################
# Flamegpu2 Visualisation Library #
###################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)
cmake_policy(SET CMP0079 NEW)

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
        add_subdirectory(${VISUALISATION_ROOT_ABS} ${CMAKE_CURRENT_BINARY_DIR}/_deps/flamegpu2_visualiser-build EXCLUDE_FROM_ALL)
        # Set locally and for parent scope, which are mutually exclusive
        set(VISUALISATION_BUILD ${flamegpu2_visualiser_BINARY_DIR} CACHE INTERNAL "flamegpu2_visualiser_BINARY_DIR")
    else()
        # Send a fatal error if the visualstion root passed is invalid.
        message(FATAL_ERROR "Invalid VISUALISATION_ROOT '${VISUALISATION_ROOT}'.\nVISUALISATION_ROOT must be a valid directory containing '${VISUALISATION_INCLUDE_HEADER_FILE}'")
    endif()

else()
    # Otherwise download.
    FetchContent_Declare(
        flamegpu2_visualiser
        GIT_REPOSITORY https://github.com/FLAMEGPU/FLAMEGPU2-visualiser.git
        GIT_TAG        master
        GIT_PROGRESS   ON
        # UPDATE_DISCONNECTED   ON
        )
        FetchContent_GetProperties(flamegpu2_visualiser)
    if(NOT flamegpu2_visualiser_POPULATED)
        FetchContent_Populate(flamegpu2_visualiser)
    
        add_subdirectory(${flamegpu2_visualiser_SOURCE_DIR} ${flamegpu2_visualiser_BINARY_DIR} EXCLUDE_FROM_ALL)
        
        # Set locally and for parent scope, which are mutually exclusive
        set(VISUALISATION_ROOT ${flamegpu2_visualiser_SOURCE_DIR} CACHE INTERNAL "flamegpu2_visualiser_SOURCE_DIR")
        set(VISUALISATION_BUILD ${flamegpu2_visualiser_BINARY_DIR} CACHE INTERNAL "flamegpu2_visualiser_BINARY_DIR")
    endif()
endif()
