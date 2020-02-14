# CMake module to find NVTX headers/library
# This is currently quite simple, could be expanded to support user-provided hinting?
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

# CMake Native CUDA support doesn't provide the raw directory, only include

# Attempt to find nvToolsExt.h containing directory
find_path(NVTX_INCLUDE_DIRS
	NAMES
		nvToolsExt.h
        nvtx3/nvToolsExt.h
    PATHS
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES
		include
		include/nvtx3
	)

# Find the directory containing the dynamic library
find_library(NVTX_LIBRARIES
	NAMES 
		libnvToolsExt.so
		nvToolsExt64_1
		nvToolsExt32_1 
	PATHS 
		${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
	PATH_SUFFIXES
		lib
		lib64
		lib/Win32
		lib/x64
)

# Apply standard cmake find package rules / variables. I.e. QUIET, NVTX_FOUND etc.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIRS NVTX_LIBRARIES)

# Set returned values as advanced?
mark_as_advanced(NVTX_INCLUDE_DIRS NVTX_LIBRARIES)
