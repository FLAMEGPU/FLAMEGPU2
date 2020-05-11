# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

project(FLAMEGPU2 LANGUAGES NONE)
set(FLAMEGPU_PROJECT_VERSION "2.0.0")

# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL ""  FORCE)

# See if the minimum CUDA version is available. If not, only enable documentation building.
set(MINIMUM_SUPPORTED_CUDA_VERSION 9.0)
include(CheckLanguage)
# See if CUDA is available
check_language(CUDA)
# If so, enable CUDA to check the version.
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()
# If CUDA is not available, or the minimum version is too low only build the docs.
if(NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
    if(NOT CMAKE_CUDA_COMPILER)
        message(STATUS "Documentation-only build: CUDA toolkit required for library compilation.")
    else()
        message(STATUS "Documentation-only build: CUDA ${MINIMUM_SUPPORTED_CUDA_VERSION} or greater is required for library compilation.")
    endif()
    #Not able to build code, so just make docs    
    get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
    include(./cmake/doxygen.cmake)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# include for dependent modules
include(CMakeDependentOption)

# Option to enable/disable building the static library
option(BUILD_FLAMEGPU2 "Enable building FLAMEGPU2 library" ON)

# Option to enable building all examples
option(BUILD_ALL_EXAMPLES "Enable building examples" ON)


# Options to enable building individual examples, if BUILD_ALL_EXAMPLES is off.
option(BUILD_EXAMPLE_HOST_FUNCTIONS "Enable building examples/host_functions" OFF)
option(BUILD_EXAMPLE_GAME_OF_LIFE "Enable building examples/game_of_life" OFF)
option(BUILD_EXAMPLE_CIRCLES_BRUTE_FORCE "Enable building examples/circles_brute_force" OFF)
option(BUILD_EXAMPLE_CIRCLES_SPATIAL_3D "Enable building examples/circles_spatial_3d" OFF)
option(BUILD_EXAMPLE_RTC_EXAMPLE "Enable building examples/rtc_example" OFF)
option(BUILD_SWIG_PYTHON "Enable python bindings via SWIG" OFF)
cmake_dependent_option(BUILD_SWIG_PYTHON_VIRTUALENV "Enable building of SWIG Python Virtual Env for Python Testing" OFF
                       "BUILD_SWIG_PYTHON" OFF)

# Option to enable/disable building the static library
option(VISUALISATION "Enable visualisation support" OFF)


# Option to enable/disable tests.
option(BUILD_TESTS "Enable building tests" OFF)

# Option to enable/disable NVTX markers for improved profiling
option(USE_NVTX "Build with NVTX markers enabled" OFF)

# Control target CUDA_ARCH to compile for
SET(CUDA_ARCH "${CUDA_ARCH}" CACHE STRING "List of CUDA Architectures to target. E.g. 61;70" FORCE)

# Enable the CXX compiler required for feature detection.
enable_language(CXX)

# If the CXX compiler is GNU, it needs to be >= 6 to build the library and >= 7 to build the tests. This will block certain cuda compilers.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # g++-6 is required - 5.5 is a broken compiler, lowever versions untested.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6)        
        message(FATAL_ERROR "g++ version >= 6 is required")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
        if (BUILD_TESTS)
            message(WARNING "  g++ < 7 is incompatible with googletest when using CUDA.\n  Setting BUILD_TESTS OFF.")
            set(BUILD_TESTS OFF)
        endif()
    endif()
endif()

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
  # Add a custom target to lint all child projects. Dependencies are specified in child projects.
  add_custom_target(all_lint)
  # Don't trigger this target on ALL_BUILD or Visual Studio 'Rebuild Solution'
  set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_ALL TRUE)
  # set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif()

# Add the library building subdirectory
if(BUILD_FLAMEGPU2)
    add_subdirectory(src "${PROJECT_BINARY_DIR}/FLAMEGPU2")
    # Set as startup project
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT flamegpu2)
endif()

# Add each example


if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_RTC_EXAMPLE)
    add_subdirectory(examples/rtc_example)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_HOST_FUNCTIONS)
    add_subdirectory(examples/host_functions)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_GAME_OF_LIFE)
    add_subdirectory(examples/game_of_life)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_CIRCLES_BRUTE_FORCE)
    add_subdirectory(examples/circles_bruteforce)
endif()

if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_CIRCLES_SPATIAL_3D)
    add_subdirectory(examples/circles_spatial3D)
endif()

# Add the tests directory (if required)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_SWIG_PYTHON)
    add_subdirectory(swig)
endif()

# Put all_lint within Lint filter
CMAKE_SET_TARGET_FOLDER(all_lint "Lint")


# message("Generated with CMAKE_BUILD_TYPE types: ${CMAKE_BUILD_TYPE}")
# message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
