message(STATUS "-----Configuring Project: ${PROJECT_NAME}-----")
include_guard(DIRECTORY)

# Policy to enable use of separate device link options, introduced in CMake 3.18
cmake_policy(SET CMP0105 NEW)

# Add custom modules directory
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

# Ensure this is not an in-source build
# This might be a little aggressive to go in comon.
include(${FLAMEGPU_ROOT}/cmake/OutOfSourceOnly.cmake)

# include CUDA_ARCH processing code.
# Uses -DCUDA_ARCH values (and modifies if appropriate). 
# Adds -gencode argumetns to cuda compiler options
# Adds -DMIN_COMPUTE_CAPABILITY=VALUE compiler defintions for C, CXX and CUDA 
include(${CMAKE_CURRENT_LIST_DIR}/cuda_arch.cmake)

# Ensure that other dependencies are downloaded and available. 
# As flamegpu is a static library, linking only only occurs at consumption not generation, so dependent targets must also know of PRIVATE shared library dependencies such as tinyxml2 and rapidjson, as well any intentionalyl public dependencies (for include dirs)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Thrust.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Jitify.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/Tinyxml2.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/dependencies/rapidjson.cmake)
if(USE_GLM)
    include(${CMAKE_CURRENT_LIST_DIR}/dependencies/glm.cmake)
endif()

# Common rules for other cmake files
# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)
# Option to promote compilation warnings to error, useful for strict CI
option(WARNINGS_AS_ERRORS "Promote compilation warnings to errors" OFF)
# Option to group CMake generated projects into folders in supported IDEs
option(CMAKE_USE_FOLDERS "Enable folder grouping of projects in IDEs." ON)
mark_as_advanced(CMAKE_USE_FOLDERS)

# Include files which define target specific functions.
include(${CMAKE_CURRENT_LIST_DIR}/warnings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cxxstd.cmake)

# Set a default build type if not passed
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(${GENERATOR_IS_MULTI_CONFIG})
    # CMAKE_CONFIGURATION_TYPES defaults to something platform specific
    # Therefore can't detect if user has changed value and not reset it
    # So force "Debug;Release"
    # set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE INTERNAL
        # "Choose the types of build, options are: Debug Release." FORCE)#
else()
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
if(USE_NVTX)
    # Find the nvtx library using custom cmake module, providing imported targets
    # Do not use CUDA::nvToolsExt as this always uses NVTX1 not 3.
    find_package(NVTX)
    # If the targets were not found, emit a warning 
    if(NOT TARGET NVTX::nvtx)
        # If not found, emit a warning and continue without NVTX
        message(WARNING "NVTX could not be found. Proceeding with USE_NVTX=OFF")
        if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
            SET(USE_NVTX "OFF" PARENT_SCOPE)
        endif()
    endif()
endif(USE_NVTX)

# Set the minimum supported cuda version, if not already set. Currently duplicated due to docs only build logic.
# CUDA 10.0 is the current minimum working but deprecated verison, which will be removed.
if(NOT DEFINED MINIMUM_CUDA_VERSION)
    set(MINIMUM_CUDA_VERSION 10.0)
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

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
    # Create the all_lint meta target if it does not exist
    if(NOT TARGET all_lint)
        add_custom_target(all_lint)
        set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
    # Define a cmake function for adding a new lint target.
    function(new_linter_target NAME SRC)
        cmake_parse_arguments(
            NEW_LINTER_TARGET
            ""
            ""
            "EXCLUDE_FILTERS"
            ${ARGN})
        # Don't lint external files
        list(FILTER SRC EXCLUDE REGEX "^${FLAMEGPU_ROOT}/externals/.*")
        # Don't lint user provided list of regular expressions.
        foreach(EXCLUDE_FILTER ${NEW_LINTER_TARGET_EXCLUDE_FILTERS})
            list(FILTER SRC EXCLUDE REGEX "${EXCLUDE_FILTER}")
        endforeach()

        # Only lint accepted file type extensions h++, hxx, cuh, cu, c, c++, cxx, cc, hpp, h, cpp, hh
        list(FILTER SRC INCLUDE REGEX ".*\\.(h\\+\\+|hxx|cuh|cu|c|c\\+\\+|cxx|cc|hpp|h|cpp|hh)$")

        # Build a list of arguments to pass to CPPLINT
        LIST(APPEND CPPLINT_ARGS "")

        # Specify output format for msvc highlighting
        if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            LIST(APPEND CPPLINT_ARGS "--output" "vs7")
        endif()
        # Set the --repository argument if included as a sub project.
        if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
            # Use find the repository root via git, to pass to cpplint.
            execute_process(COMMAND git rev-parse --show-toplevel
            WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
            RESULT_VARIABLE git_repo_found
            OUTPUT_VARIABLE abs_repo_root
            OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(git_repo_found EQUAL 0)
                LIST(APPEND CPPLINT_ARGS "--repository=${abs_repo_root}")
            endif()
        endif()
        # Add the lint_ target
        add_custom_target(
            "lint_${PROJECT_NAME}"
            COMMAND ${CPPLINT} ${CPPLINT_ARGS}
            ${SRC}
        )

        # Don't trigger this target on ALL_BUILD or Visual Studio 'Rebuild Solution'
        set_target_properties("lint_${NAME}" PROPERTIES EXCLUDE_FROM_ALL TRUE)
        # Add the custom target as a dependency of the global lint target
        if(TARGET all_lint)
            add_dependencies(all_lint lint_${NAME})
        endif()
        # Put within Lint filter
        if (CMAKE_USE_FOLDERS)
            set_property(GLOBAL PROPERTY USE_FOLDERS ON)
            set_property(TARGET "lint_${PROJECT_NAME}" PROPERTY FOLDER "Lint")
        endif ()
    endfunction()
else()
    # Don't create this message multiple times
    if(NOT COMMAND add_flamegpu_executable)
        message( 
            " cpplint: NOT FOUND!\n"
            " Lint projects will not be generated.\n"
            " Please install cpplint as described on https://pypi.python.org/pypi/cpplint.\n"
            " In most cases command 'pip install --user cpplint' should be sufficient.")
        function(new_linter_target NAME SRC)
        endfunction()
    endif()
endif()

# Define a function which can be used to set common compiler options for a target
# We do not want to force these options on end users (although they should be used ideally), hence not just public properties on the library target
# Function to suppress compiler warnings for a given target
function(CommonCompilerSettings)
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
        message( FATAL_ERROR "function(CommonCompilerSettings): 'TARGET' argument required")
    elseif(NOT TARGET ${CCS_TARGET} )
        message( FATAL_ERROR "function(CommonCompilerSettings): TARGET '${CCS_TARGET}' is not a valid target")
    endif()

    # Add device debugging symbols to device builds of CUDA objects
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>")
    # Ensure DEBUG and _DEBUG are defined for Debug builds
    target_compile_definitions(${CCS_TARGET} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:DEBUG>)
    target_compile_definitions(${CCS_TARGET} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:_DEBUG>)
    # Enable -lineinfo for Release builds, for improved profiling output.
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>")

    # Set an NVCC flag which allows host constexpr to be used on the device.
    target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

    # Prevent windows.h from defining max and min.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_definitions(${CCS_TARGET} PRIVATE NOMINMAX)
    endif()

    # Pass the SEATBELTS macro, which when set to off/0 (for non debug builds) removes expensive operations.
    if (SEATBELTS)
        # If on, all build configs have  seatbelts
        target_compile_definitions(${CCS_TARGET} PRIVATE SEATBELTS=1)
    else()
        # Id off, debug builds have seatbelts, non debug builds do not.
        target_compile_definitions(${CCS_TARGET} PRIVATE $<IF:$<CONFIG:Debug>,SEATBELTS=1,SEATBELTS=0>)
    endif()

    # MSVC handling of SYSTEM for external includes.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10)
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
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2" AND USE_NVCC_THREADS AND DEFINED NVCC_THREADS AND NVCC_THREADS GREATER_EQUAL 0)
        target_compile_options(${CCS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads ${NVCC_THREADS}>")
    endif()

endfunction()

# Function to mask some of the steps to create an executable which links against the static library
function(add_flamegpu_executable NAME SRC FLAMEGPU_ROOT PROJECT_ROOT IS_EXAMPLE)
    # @todo - correctly set PUBLIC/PRIVATE/INTERFACE for executables created with this utility function

    # Parse optional arugments.
    cmake_parse_arguments(
        ADD_FLAMEGPU_EXECUTABLE
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
    EnableFLAMEGPUCompilerWarnings(TARGET "${NAME}")
    # Apply common compiler settings
    CommonCompilerSettings(TARGET "${NAME}")
    # Set the cuda gencodes, potentially using the user-provided CUDA_ARCH
    SetCUDAGencodes(TARGET "${NAME}")
            
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
    if (VISUALISATION)
        # Copy DLLs
        # @todo clean this up. It would be much better if it were dynamic based on the visualisers's runtime dependencies too.
        if(WIN32)
            # sdl
            # if(NOT sdl2_FOUND)
                # Force finding this is disabled, as the cmake vars should already be set.
                # set(SDL2_DIR ${VISUALISATION_BUILD}/)
                # mark_as_advanced(FORCE SDL2_DIR)
                # find_package(SDL2 REQUIRED)
            # endif()
            add_custom_command(TARGET "${NAME}" POST_BUILD     # Adds a post-build event to MyTest
                COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
                    "${SDL2_RUNTIME_LIBRARIES}"                # <--this is in-file
                    $<TARGET_FILE_DIR:${NAME}>)                # <--this is out-file path
            # glew
            # if(NOT glew_FOUND)
                # Force finding this is disabled, as the cmake vars should already be set.
                # set(GLEW_DIR ${VISUALISATION_BUILD}/glew)
                # mark_as_advanced(FORCE GLEW_DIR)
                # find_package(GLEW REQUIRED)
            # endif()
            add_custom_command(TARGET "${NAME}" POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GLEW_RUNTIME_LIBRARIES}"
                    $<TARGET_FILE_DIR:${NAME}>)
            # DevIL
            # if(NOT devil_FOUND)
                # Force finding this is disabled, as the cmake vars should already be set.
                # set(DEVIL_DIR ${VISUALISATION_BUILD}/devil)
                # mark_as_advanced(FORCE DEVIL_DIR)
                # find_package(DEVIL REQUIRED NO_MODULE)
            # endif()
            add_custom_command(TARGET "${NAME}" POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${IL_RUNTIME_LIBRARIES}
                    $<TARGET_FILE_DIR:${NAME}>)
        endif()
        # @todo - this could be inherrited instead? 
        target_compile_definitions(${NAME} PRIVATE VISUALISATION)
    endif()

    # Flag the new linter target and the files to be linted, and pass optional exclusions filters (regex)
    new_linter_target(${NAME} "${SRC}" EXCLUDE_FILTERS "${ADD_FLAMEGPU_EXECUTABLE_LINT_EXCLUDE_FILTERS}")
    
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
        CMAKE_SET_TARGET_FOLDER(${NAME} "Examples")
    endif()
endfunction()

#-----------------------------------------------------------------------
# a macro that only sets the FOLDER target property if it's
# "appropriate"
# Borrowed from cmake's own CMakeLists.txt
#-----------------------------------------------------------------------
macro(CMAKE_SET_TARGET_FOLDER tgt folder)
  if(CMAKE_USE_FOLDERS)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    if(TARGET ${tgt}) # AND MSVC # AND MSVC stops all lint from being set with folder
      set_property(TARGET "${tgt}" PROPERTY FOLDER "${folder}")
    endif()
  else()
    set_property(GLOBAL PROPERTY USE_FOLDERS OFF)
  endif()
endmacro()