###################################
# CCCL (Thrust, CUB and libcucxx) #
###################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

# Set the minimum supported CCCL version, and the version to fetch
# using find_package(version) means it's up to CCCL's cmake to determine if newer versions are compatible, but this will likely need changing for CUDA 13, when CCCL is planned to have a major version bump (and drop CUDA 11 support).
set(MIN_REQUIRED_CCCL_VERSION 3.0.0)
set(CCCL_DOWNLOAD_TAG v3.0.1)

# If we have already found CCCL (and required sub-targets), don't try to find it once again.
# find_package(CCCL) is not very quiet even with QUIET passed.
if (TARGET CCCL::CCCL AND TARGET CCCL::CUB AND TARGET CCCL::Thrust AND TARGET CCCL::libcudacxx)
    if(NOT "${FOUND_CCCL_VERSION}" STREQUAL "" AND "${FOUND_CCCL_VERSION}" VERSION_LESS "${MIN_REQUIRED_CCCL_VERSION}")
        message(WARNING "Most recently found version of CCCL is insufficient (${FOUND_CCCL_VERSION} < ${MIN_REQUIRED_CCCL_VERSION}). You may need to re-generate your CMakeCache")
    endif()
    return()
endif()


# Use the FindCUDATooklit package (CMake > 3.17) to get the CUDA version and CUDA include directories for cub/thrust location hints
find_package(CUDAToolkit REQUIRED)

# Quietly find CCCL, to check if the version included with CUDA (if CCCL) is sufficiently new.
# Using CCCL avoids complex cub/thrust version workarounds previously required.
# However we cannot find thrust due to a missing guard in CCCL's cmake config file, and cannot find cub without finding libcudacxx, so just find libcudacxx quietly.
# The fix for this was first included in the 2.3.2 release
find_package(CCCL ${MIN_REQUIRED_CCCL_VERSION} QUIET COMPONENTS libcudacxx CONFIG HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_DIR}/cmake)

# If CCCL was found, find it again but loudly (with all components)
if(CCCL_FOUND)
    # Find the packages again but less quietly (and include all components)
    find_package(CCCL ${MIN_REQUIRED_CCCL_VERSION} REQUIRED CONFIG COMPONENTS HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_LIBRARY_DIR}/cmake)
    # Store the CCCL_VERSION in the cache (as advanced) for quick checking on re-runs?
    set(FOUND_CCCL_VERSION "${CCCL_VERSION}" CACHE STRING "Most recently found version of CCCL")
    mark_as_advanced(FETCHCONTENT_QUIET)
# If CCCL does need downloading, fetch it and find it (no need to add_subdirectory)
else()
    # Declare information about where and what we want from cccl.
    # Specify an invalid SOURCE_SUBDIR to prevent FetchContent_MakeAvailable from adding the CMakeLists.txt
    FetchContent_Declare(
        cccl
        GIT_REPOSITORY https://github.com/NVIDIA/CCCL.git
        GIT_TAG        ${CCCL_DOWNLOAD_TAG}
        GIT_SHALLOW    0 # @todo - set this back to 1.
        GIT_PROGRESS   ON
        # UPDATE_DISCONNECTED   ON
        SOURCE_SUBDIR "do_not_use_add_subirectory"
    )
    # Fetch the content if required.
    FetchContent_GetProperties(cccl)
    message(STATUS "Fetching CCCL ${CCCL_DOWNLOAD_TAG}")
    FetchContent_MakeAvailable(cccl)
    # Use find_package for CCLL, only looking for the fetched version.
    # This creates a non-system target due to nvcc magic to avoid the cuda toolkit version being used instead, so warnings are not suppressible without push/pop macros.
    find_package(CCCL REQUIRED CONFIG
        PATHS "${cccl_SOURCE_DIR}"
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_PACKAGE_REGISTRY
        NO_CMAKE_SYSTEM_PATH)

    # Mark some CACHE vars as advanced for a cleaner CMake GUI
    mark_as_advanced(FETCHCONTENT_QUIET)
    mark_as_advanced(FETCHCONTENT_BASE_DIR)
    mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
    mark_as_advanced(FETCHCONTENT_SOURCE_DIR_CCCL)
    mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_CCCL)
endif()

# Unset temporary variables
unset(MIN_REQUIRED_CCCL_VERSION)
unset(CCCL_DOWNLOAD_TAG)

# Mark some CACHE vars as advanced for a cleaner CMake GUI
mark_as_advanced(CCCL_DIR)
mark_as_advanced(CUB_DIR)
mark_as_advanced(Thrust_DIR)
mark_as_advanced(libcudacxx_DIR)
