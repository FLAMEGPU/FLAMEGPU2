# CMake module to find NVRTC headers/library
# This is currently quite simple, could be expanded to support user-provided hinting?
# Usage:
#    find_package( NVRTC )
#    if(NVRTC_FOUND)
#        include_directories(${NVRTC_INCLUDE_DIRS})
#        target_link_libraries(target ${NVRTC_LIBRARIES})
#    endif()
#
# Variables:
#    NVRTC_FOUND
#    NVRTC_INCLUDE_DIRS
#    NVRTC_LIBRARIES

# CMake Native CUDA support doesn't provide the raw directory, only include

# Attempt to find nvToolsExt.h containing directory
find_path(NVRTC_INCLUDE_DIRS
	NAMES
		nvrtc.h
    PATHS
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES
		include
	)

# Find the directory containing the dynamic library
find_library(NVRTC_LIBRARIES
	NAMES 
		libnvrtc.so
		nvrtc
		nvrtc.lib 
	PATHS 
		${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
	PATH_SUFFIXES
		lib
		lib64
		lib/Win32
		lib/x64
)

# Apply standard cmake find package rules / variables. I.e. QUIET, NVRTC_FOUND etc.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVRTC DEFAULT_MSG NVRTC_INCLUDE_DIRS NVRTC_LIBRARIES)

# Set returned values as advanced?
mark_as_advanced(NVRTC_INCLUDE_DIRS NVRTC_LIBRARIES)
