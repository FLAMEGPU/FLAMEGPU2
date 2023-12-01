# Minimum CMake version 3.18 for CUDA --std=c++17 
cmake_minimum_required(VERSION 3.18...3.25 FATAL_ERROR)

# Include and call some CMake to record initial state of CMAKE_CUDA_ARCHITECTURES for later use
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CUDAArchitectures.cmake)
flamegpu_init_cuda_architectures()

# Declare the project with no languages
project(FLAMEGPU LANGUAGES NONE)

# Detect if this is the top-level project or if it has been included by add_subdirectory. PROJECT_IS_TOP_LEVEL requires CMake 3.21
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(FLAMEGPU_PROJECT_IS_TOP_LEVEL ON)
else()
  set(FLAMEGPU_PROJECT_IS_TOP_LEVEL OFF)
endif()

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
if(NOT FLAMEGPU_CheckCompilerFunctionality_RESULT)
    set(DOCUMENTATION_ONLY_BUILD ON)      
    message(STATUS "Documentation-only build: due to compiler compatability version. See prior warnings.")
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
    # Set CMAKE_CUDA_ARCHITECTURES correctly, now CUDA is enabled.
    flamegpu_set_cuda_architectures()
endif()

# Set the minimum supported CUDA VERSION. 
set(MINIMUM_SUPPORTED_CUDA_VERSION 11.2)

# If the CUDA compiler is too old, trigger a docs only build.
if(NOT DOCUMENTATION_ONLY_BUILD AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
    set(DOCUMENTATION_ONLY_BUILD ON)
    message(STATUS "Documentation-only build: CUDA ${MINIMUM_SUPPORTED_CUDA_VERSION} or greater is required for compilation.")
endif()

# If CUDA is not available, or the minimum version is too low only build the docs.
if(DOCUMENTATION_ONLY_BUILD)
    # Not able to build code, so just make docs    
    include(./cmake/dependencies/doxygen.cmake)
    if(${FLAMEGPU_BUILD_API_DOCUMENTATION})
        flamegpu_create_doxygen_target("${FLAMEGPU_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "")
    endif()
    return()
endif()

# include for dependent modules
include(CMakeDependentOption)

# Option to enable building all examples, defaults to ON if FLAMEGPU is the top level cmake, else OFF
cmake_dependent_option(FLAMEGPU_BUILD_ALL_EXAMPLES "Enable building all FLAMEGPU examples" ON "FLAMEGPU_PROJECT_IS_TOP_LEVEL" OFF)

# Options to enable building individual examples, if FLAMEGPU_BUILD_ALL_EXAMPLES is off. 
# Dependent options hide these from the CMake GUI if FLAMEGPU_BUILD_ALL_EXAMPLES is on, or if it is not the top level project
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_BOIDS_BRUTEFORCE "Enable building examples/cpp/boids_bruteforce" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_BOIDS_SPATIAL3D "Enable building examples/cpp/boids_spatial3D" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_RTC_BOIDS_BRUTEFORCE "Enable building examples/cpp_rtc/boids_bruteforce" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_RTC_BOIDS_SPATIAL3D "Enable building examples/cpp_rtc/boids_spatial3D" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_CIRCLES_BRUTEFORCE "Enable building examples/cpp/circles_bruteforcespatial3D" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_CIRCLES_SPATIAL3D "Enable building examples/cpp/circles_spatial3D" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_GAME_OF_LIFE "Enable building examples/cpp/game_of_life" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_HOST_FUNCTIONS "Enable building examples/cpp/host_functions" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_ENSEMBLE "Enable building examples/cpp/ensemble" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_SUGARSCAPE "Enable building examples/cpp/sugarscape" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)
cmake_dependent_option(FLAMEGPU_BUILD_EXAMPLE_DIFFUSION "Enable building examples/cpp/diffusion" OFF "FLAMEGPU_PROJECT_IS_TOP_LEVEL; NOT FLAMEGPU_BUILD_ALL_EXAMPLES" OFF)

option(FLAMEGPU_BUILD_PYTHON "Enable python bindings via SWIG" OFF)

# Option to enable/disable tests.
option(FLAMEGPU_BUILD_TESTS "Enable building tests" OFF)

# Option to enable the development tests target, test_dev. This is independant from FLAMEGPU_BUILD_TESTS
option(FLAMEGPU_BUILD_TESTS_DEV "Enable building test_dev" OFF)

# If a mutli-config generator is beign used, and swig / python bindings are enabled, then CMake must be >= 3.20 not >= 3.18 due to cmake limitations.
if(FLAMEGPU_BUILD_PYTHON AND "${CMAKE_VERSION}" VERSION_LESS "3.20")
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(${isMultiConfig})
    # Multiconfig generators (visual studio, eclipse, ninja multi config) do not support genex in byproducts/outptus, but these are required.
    message(FATAL_ERROR "CMake >= 3.20 is required for ${PYTHON_MODULE_NAME} when using a multi-config generator such as '${CMAKE_GENERATOR}'. Please upgrade your CMake, or use a single-config geneator if possible.")
  endif()
  unset(isMultiConfig)
endif()

include(${FLAMEGPU_ROOT}/cmake/cpplint.cmake)
flamegpu_create_all_lint_target()

# Add the library building subdirectory
add_subdirectory(src "${PROJECT_BINARY_DIR}/FLAMEGPU")
# Set as startup project
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT flamegpu)

# Add each example
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_BOIDS_BRUTEFORCE)
    add_subdirectory(examples/cpp/boids_bruteforce)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_BOIDS_SPATIAL3D)
    add_subdirectory(examples/cpp/boids_spatial3D)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_RTC_BOIDS_BRUTEFORCE)
    add_subdirectory(examples/cpp_rtc/boids_bruteforce)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_RTC_BOIDS_SPATIAL3D)
    add_subdirectory(examples/cpp_rtc/boids_spatial3D)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_CIRCLES_BRUTEFORCE)
    add_subdirectory(examples/cpp/circles_bruteforce)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_CIRCLES_SPATIAL3D)
    add_subdirectory(examples/cpp/circles_spatial3D)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_GAME_OF_LIFE)
    add_subdirectory(examples/cpp/game_of_life)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_HOST_FUNCTIONS)
    add_subdirectory(examples/cpp/host_functions)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_ENSEMBLE)
    add_subdirectory(examples/cpp/ensemble)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_SUGARSCAPE)
    add_subdirectory(examples/cpp/sugarscape)
endif()
if(FLAMEGPU_BUILD_ALL_EXAMPLES OR FLAMEGPU_BUILD_EXAMPLE_DIFFUSION)
    add_subdirectory(examples/cpp/diffusion)
endif()
# Add the tests directory (if required)
if(FLAMEGPU_BUILD_TESTS OR FLAMEGPU_BUILD_TESTS_DEV)
    # Enable Ctest
    enable_testing()
    # Add the tests subdirectory
    add_subdirectory(tests)
endif()

if(FLAMEGPU_BUILD_PYTHON)
    add_subdirectory(swig)
endif()
