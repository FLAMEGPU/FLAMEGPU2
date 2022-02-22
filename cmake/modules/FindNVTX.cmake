# CMake module to find NVTX headers/library
# Finds NVTX 3 for CUDA >= 10.0, which is header only
# Finds NVTX 1 for CUDA <  10.0, which is header + library. 
#
#
# Usage:
#    find_package( NVTX )
#    if(NVTX_FOUND)
#        include_directories(${NVTX_INCLUDE_DIRS})
#        target_link_libraries(target ${NVTX_LIBRARIES})
#    endif()
#
# Variables:
#    NVTX_FOUND
#    NVTX_INCLUDE_DIRS
#    NVTX_LIBRARIES
#    NVTX_VERSION
#
# Manually specify NVRTC paths via -DNVTX_ROOT=/path/to/cuda/install/location

# CMake Native CUDA support doesn't provide the raw directory, only include
# Note that CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES and CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES are not set for visual studio generators for CMAKE < 3.17, so this relies on searching the users PATH.

# Search Path heirarchy order is <PackageName>_ROOT >  NVRTC_INCLUDE_DIRS / NVRTC_LIBRARIES > cmake environment variables > HINTS > system environment variables > platform files for current system > PATHS

include(FindPackageHandleStandardArgs)

# Search for the nvtx3 header if CUDA >= 10.0
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.0")
    # Find the nvtx header, specifically searching for the v3 header only (in the nvtx3 subdir). There are edge cases where its possible that the v2 header would be found instead.
    find_path(NVTX_INCLUDE_DIRS
        NAMES
            nvToolsExt.h
        HINTS
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        PATHS
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        PATH_SUFFIXES
            include/nvtx3
            nvtx3
        )

    if(NVTX_INCLUDE_DIRS)
        # Get the NVTX version from the headerfile, by reading it and regex matching
        file(READ "${NVTX_INCLUDE_DIRS}/nvToolsExt.h" nvtx_header_text)
        string(REGEX MATCH "define NVTX_VERSION ([0-9]+)" NVTX_VERSION_DEFINE ${nvtx_header_text})
        set(NVTX_VERSION "${CMAKE_MATCH_1}")
    endif()
    # Apply standard cmake find package rules / variables. I.e. QUIET, NVTX_FOUND etc.
    find_package_handle_standard_args(NVTX
        REQUIRED_VARS NVTX_INCLUDE_DIRS
        VERSION_VAR NVTX_VERSION
    )

    if(NVTX_FOUND)
        # Create a header only (INTERFACE) target which can be linked against to inherit include directories. Mark this as imported, because there are no build steps requred.
        add_library(NVTX::nvtx3 INTERFACE IMPORTED)
        target_include_directories(NVTX::nvtx3 INTERFACE ${NVTX_INCLUDE_DIRS})
        # Interface targets only support the version property from 
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.19")
            set_property(TARGET NVTX::nvtx3 PROPERTY VERSION ${NVTX_VERSION})
        endif()
    endif()
endif()

# @todo - find both if possible, so both targets are avialable?
# If not yet aware of NVTX, or we found V1/2 while looking for V3, make sure we find the actual V1/2
if(NOT NVTX_FOUND OR NVTX_VERSION VERSION_LESS 3)
    # Find the header file
    find_path(NVTX_INCLUDE_DIRS
        NAMES
            nvToolsExt.h
        HINTS
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        PATHS
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        PATH_SUFFIXES
            include
        )
    # Find the appropraite dynamic library - but only 64 bit.
    find_library(NVTX_LIBRARIES
        NAMES
            libnvToolsExt.so
            nvToolsExt64_1
        HINTS
            ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
        PATHS
            ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
        PATH_SUFFIXES
            lib
            lib64
            lib/x64
    )
    if(NVTX_INCLUDE_DIRS)
        # Get the NVTX version from the headerfile, by reading it and regex matching
        file(READ "${NVTX_INCLUDE_DIRS}/nvToolsExt.h" nvtx_header_text)
        string(REGEX MATCH "define NVTX_VERSION ([0-9]+)" NVTX_VERSION_DEFINE ${nvtx_header_text})
        set(NVTX_VERSION "${CMAKE_MATCH_1}")
    endif()
    # Apply standard cmake find package rules / variables. I.e. QUIET, NVTX_FOUND etc.
    find_package_handle_standard_args(NVTX
        REQUIRED_VARS NVTX_INCLUDE_DIRS
        VERSION_VAR NVTX_VERSION
    )

    if(NVTX_FOUND)
        # Create an imported target which can be linked against to inherit include directories and the shared object(s). Mark this as imported, because there are no build steps requred.
        add_library(NVTX::nvtx1 SHARED IMPORTED)
        target_include_directories(NVTX::nvtx1 INTERFACE ${NVTX_INCLUDE_DIRS})
        target_include_directories(NVTX::nvtx1 INTERFACE ${NVTX_LIBRARIES})
        set_property(TARGET NVTX::nvtx1 PROPERTY VERSION ${NVTX_VERSION})
    endif()
endif()

# Create an alias target which will alias nvtx3 if available, otherwise nvtx1
if(TARGET NVTX::nvtx3)
    add_library(NVTX::nvtx ALIAS NVTX::nvtx3)
elseif(TARGET NVTX::nvtx1)
    add_library(NVTX::nvtx ALIAS NVTX::nvtx1)
endif()
# Set returned values as advanced?
mark_as_advanced(NVTX_INCLUDE_DIRS NVTX_LIBRARIES NVTX_VERSION)
