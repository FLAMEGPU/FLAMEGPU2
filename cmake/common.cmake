message(STATUS "-----Configuring Project: ${PROJECT_NAME}-----")
include_guard(DIRECTORY)

# Policy to enable use of separate device link options, introduced in CMake 3.18
cmake_policy(SET CMP0105 NEW)

# Add custom modules directory
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

# Ensure this is not an in-source build
# This might be a little aggressive to go in comon.
include(${FLAMEGPU_ROOT}/cmake/OutOfSourceOnly.cmake)

# Ensure there are no spaces in the build directory path
include(${FLAMEGPU_ROOT}/cmake/CheckBinaryDirPathForSpaces.cmake)

# Ensure that cmake functions for handling CMAKE_CUDA_ARCHITECTURES are available
include(${FLAMEGPU_ROOT}/cmake/CUDAArchitectures.cmake)
# Emit a message once and only once per configure of the chosen architectures?
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT flamegpu_printed_cmake_cuda_architectures)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    get_directory_property(hasParent PARENT_DIRECTORY)
    if(hasParent)
        set(flamegpu_printed_cmake_cuda_architectures TRUE PARENT_SCOPE)
    endif()
    unset(hasParent)
endif()

# Ensure that other dependencies are downloaded and available. 
# As flamegpu is a static library, linking only only occurs at consumption not generation, so dependent targets must also know of PRIVATE shared library dependencies such as tinyxml2 and rapidjson, as well any intentionally public dependencies (for include dirs)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Thrust.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Jitify.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Tinyxml2.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/rapidjson.cmake)
if(FLAMEGPU_ENABLE_GLM)
    include(${CMAKE_CURRENT_LIST_DIR}/dependencies/glm.cmake)
endif()

# Common rules for other cmake files
# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)

# Option to enable/disable NVTX markers for improved profiling
option(FLAMEGPU_ENABLE_NVTX "Build with NVTX markers enabled" OFF)

# Option to enable verbose PTXAS output
option(FLAMEGPU_VERBOSE_PTXAS "Enable verbose PTXAS output" OFF)
mark_as_advanced(FLAMEGPU_VERBOSE_PTXAS)

# Option to promote compilation warnings to error, useful for strict CI
option(FLAMEGPU_WARNINGS_AS_ERRORS "Promote compilation warnings to errors" OFF)

# Option to change curand engine used for CUDA random generation
set(FLAMEGPU_CURAND_ENGINE "PHILOX" CACHE STRING "The curand engine to use. Suitable options: \"PHILOX\", \"XORWOW\", \"MRG\"")
set_property(CACHE FLAMEGPU_CURAND_ENGINE PROPERTY STRINGS PHILOX XORWOW MRG)
mark_as_advanced(FLAMEGPU_CURAND_ENGINE)

# If CUDA >= 11.2, add an option to control the use of NVCC_THREADS
set(DEFAULT_FLAMEGPU_NVCC_THREADS 2)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2)
    # The number of threads to use defaults to 2, telling the compiler to use up to 2 threads when multiple arch's are specified.
    # Setting this value to 0 would use as many threads as possible.
    # In some cases, this may increase total runtime due to excessive thread creation, and lowering the number of threads, or lowering the value of `-j` passed to cmake may be beneficial.
    if(NOT DEFINED FLAMEGPU_NVCC_THREADS)
        SET(FLAMEGPU_NVCC_THREADS "${DEFAULT_FLAMEGPU_NVCC_THREADS}" CACHE STRING "Number of concurrent threads for building multiple target architectures. 0 indicates use as many as required." FORCE)
    endif()
    mark_as_advanced(FLAMEGPU_NVCC_THREADS)
endif()

# Option to group CMake generated projects into folders in supported IDEs
option(CMAKE_USE_FOLDERS "Enable folder grouping of projects in IDEs." ON)
mark_as_advanced(CMAKE_USE_FOLDERS)

# Include files which define target specific functions.
include(${CMAKE_CURRENT_LIST_DIR}/warnings.cmake)

# Ensure that flamegpu_set_target_folder is available
include(${CMAKE_CURRENT_LIST_DIR}/SetTargetFolder.cmake)

# Set a default build type if not passed
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT ${GENERATOR_IS_MULTI_CONFIG})
    if(NOT CMAKE_BUILD_TYPE)
        set(default_build_type "Release")
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING 
            "Choose the type of build, options are: Release, Debug, RelWithDebInfo, MinSizeRel or leave the value empty for the default." FORCE)
    endif()
endif()

# Ask Cmake to output compile_commands.json (if supported). This is useful for vscode include paths, clang-tidy/clang-format etc
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "Control the output of compile_commands.json")

# Use the FindCUDATooklit package (CMake > 3.17) to find other parts of the cuda toolkit not provided by the CMake language support
find_package(CUDAToolkit REQUIRED)

# Control how we link against the cuda runtime library (CMake >= 3.17)
# We may wish to use static or none instead, subject to python library handling.
set(CMAKE_CUDA_RUNTIME_LIBRARY shared)

# Ensure the cuda driver API is available, and save it to the list of link targets.
if(NOT TARGET CUDA::cuda_driver)
    message(FATAL_ERROR "CUDA::cuda_driver is required.")
endif()

# Ensure the nvrtc is available, and save it to the list of link targets.
if(NOT TARGET CUDA::nvrtc)
    message(FATAL_ERROR "CUDA::nvrtc is required.")
endif()

# Ensure that jitify is available. Must be available at binary link time due to flamegpu being a static library. This check may be redundant.
if(NOT TARGET Jitify::jitify)
    message(FATAL_ERROR "Jitify is a required dependency")
endif()

# Ensure that 

# @todo - why do we not have to link against curand? Is that only required for the host API? Use CUDA::curand if required.

# If NVTX is enabled, find the library and update variables accordingly. 
if(FLAMEGPU_ENABLE_NVTX)
    # Find the nvtx library using custom cmake module, providing imported targets
    # Do not use CUDA::nvToolsExt as this always uses NVTX1 not 3.
    # See https://gitlab.kitware.com/cmake/cmake/-/issues/21377
    find_package(NVTX)
    # If the targets were not found, emit a warning 
    if(NOT TARGET NVTX::nvtx)
        # If not found, emit a warning and continue without NVTX
        message(WARNING "NVTX could not be found. Proceeding with FLAMEGPU_ENABLE_NVTX=OFF")
        if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
            SET(FLAMEGPU_ENABLE_NVTX "OFF" PARENT_SCOPE)
        endif()
    endif()
endif(FLAMEGPU_ENABLE_NVTX)

# Set the minimum supported cuda version, if not already set. Currently duplicated due to docs only build logic.
# CUDA 11.0 is current minimum cuda version, and the minimum supported
if(NOT DEFINED MINIMUM_CUDA_VERSION)
    set(MINIMUM_CUDA_VERSION 11.0)
    # Require a minimum cuda version
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_CUDA_VERSION})
        message(FATAL_ERROR "CUDA version must be at least ${MINIMUM_CUDA_VERSION}")
    endif()
endif()
# CUDA 11.0 is the current minimum supported version.
if(NOT DEFINED MINIMUM_SUPPORTED_CUDA_VERSION)
    set(MINIMUM_SUPPORTED_CUDA_VERSION 11.0)
    # Warn on deprecated cuda version.
    # If the CUDA compiler is atleast the minimum deprecated version, but less than the minimum actually supported version, issue a dev warning.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL ${MINIMUM_CUDA_VERSION} AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
        message(DEPRECATION "Support for CUDA verisons <= ${MINIMUM_SUPPORTED_CUDA_VERSION} is deprecated and will be removed in a future release.")
    endif()
endif()

# Invlude the cpplint camake, which provides a function to create a lint target.
include(${CMAKE_CURRENT_LIST_DIR}/cpplint.cmake)


# Define a function which can be used to set common compiler options for a target
# We do not want to force these options on end users (although they should be used ideally), hence not just public properties on the library target
# Function to suppress compiler warnings for a given target
function(flamegpu_common_compiler_settings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        CCS
        ""
        "TARGET"
        ""
        ${ARGN}
    )

    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT CCS_TARGET)
        message( FATAL_ERROR "flamegpu_common_compiler_settings: 'TARGET' argument required")
    elseif(NOT TARGET ${CCS_TARGET} )
        message( FATAL_ERROR "flamegpu_common_compiler_settings: TARGET '${CCS_TARGET}' is not a valid target")
    endif()

    # Enable -lineinfo for Release builds, for improved profiling output.
    # CMAKE >=3.19 required for multivalue CONFIG:
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>,$<CONFIG:RelWithDebInfo>>>:-lineinfo>")

    # Set an NVCC flag which allows host constexpr to be used on the device.
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

    # MSVC handling of SYSTEM for external includes, present in 19.10+
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # These flags don't currently have any effect on how CMake passes system-private includes to msvc (VS 2017+)
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I")
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "/external:I")
        # VS 2017+
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/experimental:external>")
    endif()

    # Enable parallel compilation
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /MP>")
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/MP>")
    endif()

    # If CUDA 11.2+, can build multiple architectures in parallel. 
    # Note this will be multiplicative against the number of threads launched for parallel cmake build, which may lead to processes being killed, or excessive memory being consumed.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2" AND DEFINED FLAMEGPU_NVCC_THREADS)
        set(FLAMEGPU_NVCC_THREADS_INTEGER -1})
    # If its a number GE 0, use that, this is false for truthy values
        if(FLAMEGPU_NVCC_THREADS GREATER_EQUAL 0)
            set(FLAMEGPU_NVCC_THREADS_INTEGER ${FLAMEGPU_NVCC_THREADS})
        # If it is not set, use a hardcoded sensible default 2.
        elseif("${FLAMEGPU_NVCC_THREADS}" STREQUAL "")
            set(FLAMEGPU_NVCC_THREADS_INTEGER ${DEFAULT_FLAMEGPU_NVCC_THREADS})
        # Otherwise, use 1, alternativel we could fatal error here.
        else()
            set(FLAMEGPU_NVCC_THREADS_INTEGER 1)
        endif()
        if(FLAMEGPU_NVCC_THREADS_INTEGER GREATER_EQUAL 0)
            target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads ${FLAMEGPU_NVCC_THREADS_INTEGER}>")
        endif()
    endif()

    # Enable verbose ptxas output if required 
    if(FLAMEGPU_VERBOSE_PTXAS)
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas -v>")
    endif()
    
endfunction()

# Function to mask some of the steps to create an executable which links against the static library
function(flamegpu_add_executable NAME SRC FLAMEGPU_ROOT PROJECT_ROOT IS_EXAMPLE)
    # @todo - correctly set PUBLIC/PRIVATE/INTERFACE for executables created with this utility function

    # Parse optional arugments.
    cmake_parse_arguments(
        FLAMEGPU_ADD_EXECUTABLE
        ""
        ""
        "LINT_EXCLUDE_FILTERS"
        ${ARGN})

    # If the library does not exist as a target, add it.
    if (NOT TARGET flamegpu)
        add_subdirectory("${FLAMEGPU_ROOT}/src" "${PROJECT_ROOT}/FLAMEGPU")
    endif()

    if(WIN32)
      # configure a rc file to set application icon
      set (FLAMEGPU_ICON_PATH ${FLAMEGPU_ROOT}/cmake/flamegpu.ico)
      set (PYFLAMEGPU_ICON_PATH ${FLAMEGPU_ROOT}/cmake/pyflamegpu.ico)
      configure_file(
        ${FLAMEGPU_ROOT}/cmake/application_icon.rc.in
        application_icon.rc
        @ONLY)
      SET(SRC ${SRC} application_icon.rc)
    endif()

    # Define which source files are required for the target executable
    add_executable(${NAME} ${SRC})

    # Set target level warnings.
    flamegpu_enable_compiler_warnings(TARGET "${NAME}")
    # Apply common compiler settings
    flamegpu_common_compiler_settings(TARGET "${NAME}")
    
    # Set C++17 using modern CMake options
    target_compile_features(${NAME} PUBLIC cxx_std_17)
    target_compile_features(${NAME} PUBLIC cuda_std_17)
    set_property(TARGET ${NAME} PROPERTY CXX_EXTENSIONS OFF)
    set_property(TARGET ${NAME} PROPERTY CUDA_EXTENSIONS OFF)
    set_property(TARGET ${NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET ${NAME} PROPERTY CUDA_STANDARD_REQUIRED ON)

    # Enable RDC for the target
    set_property(TARGET ${NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Link against the flamegpu static library target.
    target_link_libraries(${NAME} PRIVATE flamegpu)
    # Workaround for incremental rebuilds on MSVC, where device link was not being performed.
    # https://github.com/FLAMEGPU/FLAMEGPU2/issues/483
    if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
        # Provide the absolute path to the lib file, rather than the relative version cmake provides.
        target_link_libraries(${NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE:flamegpu>")
    endif()
        
    # Activate visualisation if requested
    if (FLAMEGPU_VISUALISATION)
        # Copy DLLs / other Runtime dependencies
        if(COMMAND flamegpu_visualiser_get_runtime_depenencies)
            flamegpu_visualiser_get_runtime_depenencies(vis_runtime_dependencies)
            # For each runtime dependency (dll)
            foreach(vis_runtime_dependency ${vis_runtime_dependencies})
                # Add a post build comamnd which copies the dll to the directory of the binary if needed.
                add_custom_command(
                    TARGET "${NAME}" POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${vis_runtime_dependency}"
                        $<TARGET_FILE_DIR:${NAME}>
                )
            endforeach()
            unset(vis_runtime_dependencies)
        endif()
    endif()

    # Flag the new linter target and the files to be linted, and pass optional exclusions filters (regex)
    flamegpu_new_linter_target(${NAME} "${SRC}" EXCLUDE_FILTERS "${FLAMEGPU_ADD_EXECUTABLE_LINT_EXCLUDE_FILTERS}")
    
    # Setup Visual Studio (and eclipse) filters
    #src/.h
    set(T_SRC "${SRC}")
    list(FILTER T_SRC INCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/src")
    list(FILTER T_SRC INCLUDE REGEX ".*\.(h|hpp|cuh)$")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX headers FILES ${T_SRC})
    #src/.cpp
    set(T_SRC "${SRC}")
    list(FILTER T_SRC INCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/src")
    list(FILTER T_SRC EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX src FILES ${T_SRC})
    #./.h
    set(T_SRC "${SRC}")
    list(FILTER T_SRC EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/src")
    list(FILTER T_SRC INCLUDE REGEX ".*\.(h|hpp|cuh)$")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX headers FILES ${T_SRC})
    #./.cpp
    set(T_SRC "${SRC}")
    list(FILTER T_SRC EXCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/src")
    list(FILTER T_SRC EXCLUDE REGEX ".*\.(h|hpp|cuh|rc)$")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX src FILES ${T_SRC})


    # Put within Examples filter
    if(IS_EXAMPLE)
        flamegpu_set_target_folder(${NAME} "Examples")
    endif()
endfunction()

