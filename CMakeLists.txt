# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

project(FLAMEGPU LANGUAGES NONE)

# Find the root directory
get_filename_component(FLAMEGPU_ROOT ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)

# Ensure this is not an in-source build
include(${FLAMEGPU_ROOT}/cmake/OutOfSourceOnly.cmake)

# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL ""  FORCE)

# If sufficiently new and working CXX / CUDA compilers are not available, then documentation only build is the only option.
set(DOCUMENTATION_ONLY_BUILD OFF)

# Check compiler functionailty, as there are known issues in some cases, but version checks are not always sufficient.
include(./cmake/CheckCompilerFunctionality.cmake)
# If this returned a negative result, set the docs only build. 
if(NOT CheckCompilerFunctionality_RESULT)
    set(DOCUMENTATION_ONLY_BUILD ON)      
    message(STATUS "Documentation-only build: due to Compiler compatability version. See prior warnings.")
endif()

# If the checks passed, enable CXX and CUDA languages
include(CheckLanguage)
check_language(CXX)
if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

# CUDA 10.0 is the current minimum working but deprecated verison, which will be removed.
set(MINIMUM_CUDA_VERSION 10.0)
# If the CUDA compiler is too old, trigger a docs only build.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_CUDA_VERSION})
    set(DOCUMENTATION_ONLY_BUILD ON)
    message(STATUS "Documentation-only build: CUDA ${MINIMUM_SUPPORTED_CUDA_VERSION} or greater is required for compilation.")
endif()

# CUDA 11.0 is the current minimum supported version.
set(MINIMUM_SUPPORTED_CUDA_VERSION 11.0)
# If the CUDA compiler is atleast the minimum deprecated version, but less than the minimum actually supported version, issue a dev warning.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL ${MINIMUM_CUDA_VERSION} AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
    message(DEPRECATION "Support for CUDA verisons <= ${MINIMUM_SUPPORTED_CUDA_VERSION} is deprecated and will be removed in a future release.")
endif()

# If CUDA is not available, or the minimum version is too low only build the docs.
if(DOCUMENTATION_ONLY_BUILD)
    # Not able to build code, so just make docs    
    include(./cmake/dependencies/doxygen.cmake)
    if(${BUILD_API_DOCUMENTATION})
        create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# include for dependent modules
include(CMakeDependentOption)

# Option to enable/disable building the static library
option(BUILD_FLAMEGPU "Enable building FLAMEGPU library" ON)

# Option to enable/disable building the static library
option(VISUALISATION "Enable visualisation support" OFF)

if(NOT NO_EXAMPLES)
    # Option to enable building all examples
    option(BUILD_ALL_EXAMPLES "Enable building examples" ON)

    # Options to enable building individual examples, if BUILD_ALL_EXAMPLES is off.
    option(BUILD_EXAMPLE_BOIDS_BRUTEFORCE "Enable building examples/boids_bruteforce" OFF)
    option(BUILD_EXAMPLE_BOIDS_BRUTEFORCE_DEPENDENCY_GRAPH "Enable building examples/boids_bruteforce_dependency_graph" OFF)
    option(BUILD_EXAMPLE_BOIDS_SPATIAL3D "Enable building examples/boids_spatial3D" OFF)
    option(BUILD_EXAMPLE_BOIDS_RTC_BRUTEFORCE "Enable building examples/boids_rtc_bruteforce" OFF)
    option(BUILD_EXAMPLE_BOIDS_RTC_SPATIAL3D "Enable building examples/boids_rtc_spatial3D" OFF)
    option(BUILD_EXAMPLE_CIRCLES_BRUTEFORCE "Enable building examples/circles_bruteforcespatial3D" OFF)
    option(BUILD_EXAMPLE_CIRCLES_SPATIAL3D "Enable building examples/circles_spatial3D" OFF)
    option(BUILD_EXAMPLE_GAME_OF_LIFE "Enable building examples/game_of_life" OFF)
    option(BUILD_EXAMPLE_HOST_FUNCTIONS "Enable building examples/host_functions" OFF)
    option(BUILD_EXAMPLE_ENSEMBLE "Enable building examples/ensemble" OFF)
    option(BUILD_EXAMPLE_SUGARSCAPE "Enable building examples/sugarscape" OFF)
endif()

option(BUILD_SWIG_PYTHON "Enable python bindings via SWIG" OFF)
# By default, build into a venv, otherwise it is set to ON anyway. 
cmake_dependent_option(BUILD_SWIG_PYTHON_VENV "Enable the use of a venv for swig/python installation" ON "BUILD_SWIG_PYTHON" ON)
# Add option to embed the cuda version in the python local version, for pseudo downstream releases / to differentiate wheel files.
cmake_dependent_option(BUILD_SWIG_PYTHON_LOCALVERSION "Embed CUDA version for the build in the local information" ON "BUILD_SWIG_PYTHON" ON)


# Option to enable/disable tests.
option(BUILD_TESTS "Enable building tests" OFF)

# Option to enable the development tests target, test_dev. This is independant from build_tests
option(BUILD_TESTS_DEV "Enable building test_dev" OFF)

# Option to enable/disable NVTX markers for improved profiling
option(USE_NVTX "Build with NVTX markers enabled" OFF)

# Option to enable/disable logging of dynamic RTC files to disk
option(EXPORT_RTC_SOURCES "Export RTC source files to disk at runtime" OFF)

# If CUDA >= 11.2, add an option to enable using NVCC_THREASD
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2)
    option(USE_NVCC_THREADS "Enable parallel compilation of multiple NVCC targets. See NVCC_THREADS for more control." ON)
    # The number of threads to use defaults to 2, telling the compiler to use up to 2 threads when multiple arch's are specified.
    # Setting this value to 0 would use as many threads as possible.
    # In some cases, this may increase total runtime due to excessive thread creation, and lowering the number of threads, or lowering the value of `-j` passed to cmake may be beneficial.
    if(NOT DEFINED NVCC_THREADS)
        set(NVCC_THREADS "2")
    endif()
    SET(NVCC_THREADS "${NVCC_THREADS}" CACHE STRING "Number of concurrent threads for building multiple target architectures. 0 indicates use as many as required." FORCE)
endif()

# Control target CUDA_ARCH to compile for
SET(CUDA_ARCH "${CUDA_ARCH}" CACHE STRING "List of CUDA Architectures to target. E.g. 61;70" FORCE)

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
if(BUILD_FLAMEGPU)
    add_subdirectory(src "${PROJECT_BINARY_DIR}/FLAMEGPU")
    # Set as startup project
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT flamegpu)
endif()

# Add each example
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_BOIDS_BRUTEFORCE)
    add_subdirectory(examples/boids_bruteforce)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_BOIDS_BRUTEFORCE_DEPENDENCY_GRAPH)
    add_subdirectory(examples/boids_bruteforce_dependency_graph)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_BOIDS_SPATIAL3D)
    add_subdirectory(examples/boids_spatial3D)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_BOIDS_RTC_BRUTEFORCE)
    add_subdirectory(examples/boids_rtc_bruteforce)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_BOIDS_RTC_SPATIAL3D)
    add_subdirectory(examples/boids_rtc_spatial3D)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_CIRCLES_BRUTEFORCE)
    add_subdirectory(examples/circles_bruteforce)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_CIRCLES_SPATIAL3D)
    add_subdirectory(examples/circles_spatial3D)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_GAME_OF_LIFE)
    add_subdirectory(examples/game_of_life)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_HOST_FUNCTIONS)
    add_subdirectory(examples/host_functions)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_ENSEMBLE)
    add_subdirectory(examples/ensemble)
endif()
if(BUILD_ALL_EXAMPLES OR BUILD_EXAMPLE_SUGARSCAPE)
    add_subdirectory(examples/sugarscape)
endif()
# Add the tests directory (if required)
if(BUILD_TESTS OR BUILD_TESTS_DEV)
    # g++ 7 is required for c++ tests to build.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
        message(WARNING "  g++ < 7 is incompatible with googletest when using CUDA.\n  Setting BUILD_TESTS OFF.")
        set(BUILD_TESTS OFF)
        set(BUILD_TESTS_DEV OFF)
    else()
        add_subdirectory(tests)
    endif()
endif()

if(BUILD_SWIG_PYTHON)
    add_subdirectory(swig)
endif()

# Put all_lint within Lint filter
CMAKE_SET_TARGET_FOLDER(all_lint "Lint")


# message("Generated with CMAKE_BUILD_TYPE types: ${CMAKE_BUILD_TYPE}")
# message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
