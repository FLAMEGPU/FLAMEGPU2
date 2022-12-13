###################################
# Flamegpu2 Visualisation Library #
###################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# Set the visualiser repo and tag to use unless overridden by the user.
set(DEFAULT_FLAMEGPU_VISUALISATION_GIT_VERSION "flamegpu-2.0.0-rc")
set(DEFAULT_FLAMEGPU_VISUALISATION_REPOSITORY "https://github.com/FLAMEGPU/FLAMEGPU2-visualiser.git")

# Set a VISUSLAITION_ROOT cache entry so it is available in the GUI to override the location if required
if(NOT DEFINED CACHE{FLAMEGPU_VISUALISATION_ROOT})
    set(FLAMEGPU_VISUALISATION_ROOT "" CACHE STRING "Path to local copy of the FLAMEGPU2-visualiser repository, rather than CMake-based fetching")
endif()

# Detect if the user provided the visualisation root or ot, by comparing to the fetch content source dir.
if (FLAMEGPU_VISUALISATION_ROOT)
    # @todo - we should make the visualisation package find_package() compatible, and check it exists if FLAMEGPU_VISUALISATION_ROOT is set. 
    # Look for the main visualisation header to get the abs path, but only look relative to the hints/paths, no cmake defaults (for now)
    set(FLAMEGPU_VISUALISATION_INCLUDE_HEADER_FILE include/flamegpu/visualiser/FLAMEGPU_Visualisation.h)
    find_path(FLAMEGPU_VISUALISATION_ROOT_ABS
        NAMES
            ${FLAMEGPU_VISUALISATION_INCLUDE_HEADER_FILE}
        HINTS
            ${FLAMEGPU_VISUALISATION_ROOT}
        PATHS
            ${FLAMEGPU_VISUALISATION_ROOT}
        NO_DEFAULT_PATH
    )
    # If found, use the local vis, otherwise error.
    if(FLAMEGPU_VISUALISATION_ROOT_ABS)
        # If the correct visualtiion root was found, output a successful status message
        message(STATUS "Found FLAMEGPU_VISUALISATION_ROOT: ${FLAMEGPU_VISUALISATION_ROOT_ABS} (${FLAMEGPU_VISUALISATION_ROOT})")
        # update the value to the non abs version, in local and parent scope.
        set(FLAMEGPU_VISUALISATION_ROOT "${FLAMEGPU_VISUALISATION_ROOT_ABS}")
        set(FLAMEGPU_VISUALISATION_ROOT "${FLAMEGPU_VISUALISATION_ROOT_ABS}" PARENT_SCOPE)
        # And set up the visualisation build 
        add_subdirectory(${FLAMEGPU_VISUALISATION_ROOT_ABS} ${CMAKE_CURRENT_BINARY_DIR}/_deps/flamegpu_visualiser-build EXCLUDE_FROM_ALL)
        # Set the cahce var too, to ensure it appears in the GUI.
        set(FLAMEGPU_VISUALISATION_ROOT "${FLAMEGPU_VISUALISATION_ROOT}" CACHE STRING "Path to local copy of the FLAMEGPU2-visualiser repository, rather than CMake-based fetching")

    else()
        # Send a fatal error if the visualstion root passed is invalid.
        message(FATAL_ERROR "Invalid FLAMEGPU_VISUALISATION_ROOT '${FLAMEGPU_VISUALISATION_ROOT}'.\nFLAMEGPU_VISUALISATION_ROOT must be a valid directory containing '${FLAMEGPU_VISUALISATION_INCLUDE_HEADER_FILE}'")
    endif()
else()
    # If not using a user-specified FLAMEGPU_VISUALISER_ROOT, fetch content vi CMake
    # If a FLAMEGPU_VISUALISATION_GIT_VERSION has not been defined, set it to the default option.
    if(NOT DEFINED FLAMEGPU_VISUALISATION_GIT_VERSION OR FLAMEGPU_VISUALISATION_GIT_VERSION STREQUAL "")
        set(FLAMEGPU_VISUALISATION_GIT_VERSION "${DEFAULT_FLAMEGPU_VISUALISATION_GIT_VERSION}" CACHE STRING "Git branch or tag to use for the FLAMEPGU2_visualiser")
    endif()
    mark_as_advanced(FLAMEGPU_VISUALISATION_GIT_VERSION)

    # Allow users to switch to forks with relative ease.
    if(NOT DEFINED FLAMEGPU_VISUALISATION_REPOSITORY OR FLAMEGPU_VISUALISATION_REPOSITORY STREQUAL "")
        set(FLAMEGPU_VISUALISATION_REPOSITORY "${DEFAULT_FLAMEGPU_VISUALISATION_REPOSITORY}" CACHE STRING "Remote Git Repository for the FLAMEPGU2_visualiser")
    endif()
    mark_as_advanced(FLAMEGPU_VISUALISATION_REPOSITORY)

    # Otherwise download.
    FetchContent_Declare(
        flamegpu_visualiser
        GIT_REPOSITORY ${FLAMEGPU_VISUALISATION_REPOSITORY}
        GIT_TAG        ${FLAMEGPU_VISUALISATION_GIT_VERSION}
        GIT_PROGRESS   ON
    )
    FetchContent_GetProperties(flamegpu_visualiser)
    if(NOT flamegpu_visualiser_POPULATED)
        message(STATUS "using flamegpu_visualiser ${FLAMEGPU_VISUALISATION_GIT_VERSION} from ${FLAMEGPU_VISUALISATION_REPOSITORY}")
        FetchContent_Populate(flamegpu_visualiser)
        # Add the project as a subdirectory
        add_subdirectory(${flamegpu_visualiser_SOURCE_DIR} ${flamegpu_visualiser_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()
# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FLAMEGPU_VISUALISER)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FLAMEGPU_VISUALISER) 
mark_as_advanced(FLAMEGPU_VISUALISATION_ROOT_ABS)
# Uneset some variables to avoid scope leaking.
unset(DEFAULT_FLAMEGPU_VISUALISATION_GIT_VERSION)
unset(DEFAULT_FLAMEGPU_VISUALISATION_REPOSITORY)
