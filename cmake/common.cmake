message(STATUS "-----Configuring Project: ${PROJECT_NAME}-----")
include_guard(DIRECTORY)
if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    cmake_policy(SET CMP0105 NEW) # Use separate device link options
endif()
# Add custom modules directory
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

# include CUDA_ARCH processing code.
# Uses -DCUDA_ARCH values (and modifies if appropriate). 
# Adds -gencode argumetns to CMAKE_CUDA_FLAGS
# Adds -DMIN_COMPUTE_CAPABILITY=VALUE macro to CMAKE_CC_FLAGS, CMAKE_CXX_FLAGS and CMAKE_CUDA_FLAGS.
include(${CMAKE_CURRENT_LIST_DIR}/cuda_arch.cmake)

# Ensure that other dependencies are downloaded and available. 
# This could potentially go in the src cmake.list, once headers do not include third party headers.
include(${CMAKE_CURRENT_LIST_DIR}/Thrust.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Jitify.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Tinyxml2.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/rapidjson.cmake)

# Common rules for other cmake files
# Make available lowercase 'linux'/'windows' vars (used for build dirs)
STRING(TOLOWER "${CMAKE_SYSTEM_NAME}" CMAKE_SYSTEM_NAME_LOWER)
# Don't create installation scripts (and hide CMAKE_INSTALL_PREFIX from cmake-gui)
set(CMAKE_SKIP_INSTALL_RULES TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE INTERNAL "" FORCE)
# Option to promote compilation warnings to error, useful for strict CI
option(WARNINGS_AS_ERRORS "Promote compilation warnings to errors" OFF)
# Option to group CMake generated projects into folders in supported IDEs
option(CMAKE_USE_FOLDERS "Enable folder grouping of projects in IDEs." ON)
mark_as_advanced(CMAKE_USE_FOLDERS)


# Cmake 3.16 has an issue with the order of CUDA includes when using SYSTEM for user-provided thrust. Avoid this by not using SYSTEM for cmake 3.16
set(INCLUDE_SYSTEM_FLAG SYSTEM)
if(${CMAKE_VERSION} VERSION_GREATER "3.16" AND ${CMAKE_VERSION} VERSION_LESS "3.17") 
    set(INCLUDE_SYSTEM_FLAG "")
endif()

# Set a default build type if not passed
get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(${GENERATOR_IS_MULTI_CONFIG})
    # CMAKE_CONFIGURATION_TYPES defaults to something platform specific
    # Therefore can't detect if user has changed value and not reset it
    # So force "Debug;Release;Profile"
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile" CACHE INTERNAL
        "Choose the types of build, options are: Debug Release Profile." FORCE)#
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(default_build_type "Release")
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING 
            "Choose the type of build, options are: None Debug Release Profile." FORCE)
    endif()
endif()

# Create the profile build modes, based on release
SET( CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the C++ compiler during profile builds."
    FORCE )
SET( CMAKE_C_FLAGS_PROFILE "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the C compiler during profile builds."
    FORCE )
SET( CMAKE_CUDA_FLAGS_PROFILE "${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the CUDA compiler during profile builds."
    FORCE )
SET( CMAKE_EXE_LINKER_FLAGS_PROFILE
    "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING
    "Flags used for linking binaries during profile builds."
    FORCE )
SET( CMAKE_SHARED_LINKER_FLAGS_PROFILE
    "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING
    "Flags used by the shared libraries linker during profile builds."
    FORCE )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_PROFILE
    CMAKE_C_FLAGS_PROFILE
    CMAKE_EXE_LINKER_FLAGS_PROFILE
    CMAKE_SHARED_LINKER_FLAGS_PROFILE )

    
    # If using profile build, imply NVTX
    if(CMAKE_BUILD_TYPE MATCHES "Profile")
    SET(NVTX "ON")
    endif()
    
    
# Declare variables to track extra include dirs / link dirs / link libraries
set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES)
set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES)

# NVRTC.lib/CUDA.lib

find_package(NVRTC REQUIRED)
if(NVRTC_FOUND)
    set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} "${NVRTC_INCLUDE_DIRS}")
    set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} ${NVRTC_LIBRARIES})
    # Also add the driver api
    set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} cuda)
else()
    message("nvrtc not found @todo gracefully handle this")
endif()


# If NVTX is enabled, find the library and update variables accordingly.
if(USE_NVTX)
    # Find the nvtx library using custom cmake module
    find_package(NVTX)
    # If it was found, use it.
    if(NVTX_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_NVTX=${NVTX_VERSION}")
        set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} "${NVTX_INCLUDE_DIRS}")
        if(NVTX_VERSION VERSION_LESS "3")
            set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} ${NVTX_LIBRARIES})
        endif()
    else()
        # If not found, disable.
        message("-- NVTX not available")
        SET(USE_NVTX "OFF" PARENT_SCOPE)    
    endif()
endif(USE_NVTX)

# If jitify was found, add it to the include dirs.
if(Jitify_FOUND)
    set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} "${Jitify_INCLUDE_DIRS}")
endif()

# If gcc, need to add linker flag for std::experimental::filesystem pre c++17
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} "-lstdc++fs")
endif()

# Logging for jitify compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DJITIFY_PRINT_LOG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJITIFY_PRINT_LOG")

# Set the minimum supported cuda version, if not already set. Currently duplicated due to docs only build logic.
if(NOT DEFINED MINIMUM_SUPPORTED_CUDA_VERSION)
    set(MINIMUM_SUPPORTED_CUDA_VERSION 10.0)
endif()
# Require a minimum cuda version
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
    message(FATAL_ERROR "CUDA version must be at least ${MINIMUM_SUPPORTED_CUDA_VERSION}")
endif()

# Specify some additional compiler flags
# CUDA debug symbols
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -D_DEBUG -DDEBUG")

# Lineinfo for non -G release
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

# profile specific CUDA flags.
set(CMAKE_CUDA_FLAGS_PROFILE "${CMAKE_CUDA_FLAGS_PROFILE} -lineinfo -DPROFILE -D_PROFILE")
# Addresses a cub::histogram warning
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# Enable default stream per thread for target, in-case of ensembles
# This removes implicit syncs in default stream, using it has only been tested for basic models
# It has not been tested with host functions, agent death, optional messages etc
# Hence using it is unsafe
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")    
    
# Host Compiler version specific high warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Only set W4 for MSVC, WAll is more like Wall, Wextra and Wpedantic
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    # Also suppress some unwanted W4 warnings
    # decorated name length exceeded, name was truncated
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4503")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4503")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4503")
    # 'function' : unreferenced local function has been removed
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4505")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4505")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4505")
    # unreferenced formal parameter warnings disabled - tests make use of it.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4100")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4100")
    # Suppress some VS2015 specific warnings.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4091")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4091")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4091")
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10)
        # These flags don't currently have any effect on how CMake passes system-private includes to msvc (VS 2017+)
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I")
        set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "/external:I")
        # VS 2017+
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /experimental:external")
    endif()
else()
    # Assume using GCC/Clang which Wall is relatively sane for. 
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wsign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wsign-compare")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wsign-compare")

    # CUB 1.9.10 prevents Wreorder being usable on windows. Cannot suppress via diag_suppress pragmas.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wreorder")
endif()

# Promote  warnings to errors if requested
if(WARNINGS_AS_ERRORS)
    # OS Specific flags
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /WX")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    endif()

    # Generic WError settings for nvcc
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=\"-Werror\"  -Xnvlink=\"-Werror\"")

    # If CUDA 10.2+, add all_warnings to the Werror option
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.2")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")
    endif()

    # If not msvc, add cross-execution-space-call. This is blocked under msvc by a jitify related bug (untested > CUDA 10.1): https://github.com/NVIDIA/jitify/issues/62
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call ")
    endif()

    # If not msvc, add reorder to Werror. This is blocked under msvc by cub/thrust and the lack of isystem on msvc. Appears unable to suppress the warning via diag_suppress pragmas.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror reorder ")
    endif()
endif()

# Ask the cuda frontend to include warnings numbers, so they can be targetted for suppression.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Suppress nodiscard warnings from the cuda frontend
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2809")
endif()

# Common CUDA args

# Use C++14 standard - std::make_unique is 14 not 11
# Specify using C++14 standard
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()

# Tell CUDA to use C++14 standard.
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    # This has no effect on msvc, even though CUDA supports it...
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)

    # Actually pass --std=c++14 on windows for VS 2017+ and CUDA 11+
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10 AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 ")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /std:c++14")
    endif()
endif()

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
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
        # Add custom target for linting this
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # Output in visual studio format
            add_custom_target(
                "lint_${NAME}"
                COMMAND ${CPPLINT} "--output" "vs7"
                ${SRC}
            )
        else()
            # Output in default format
            add_custom_target(
                "lint_${NAME}"
                COMMAND ${CPPLINT}
                ${SRC}
            )
        endif()
        # Don't trigger this target on ALL_BUILD or Visual Studio 'Rebuild Solution'
        set_target_properties("lint_${NAME}" PROPERTIES EXCLUDE_FROM_ALL TRUE)
        # set_target_properties("lint_${NAME}" PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE #This breaks all_lint in visual studio
        # Add the custom target as a dependency of the global lint target
        if(TARGET all_lint)
            add_dependencies(all_lint lint_${NAME})
        endif()
        # Put within Lint filter
        CMAKE_SET_TARGET_FOLDER("lint_${NAME}" "Lint")
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

# Function to mask some of the steps to create an executable which links against the static library
function(add_flamegpu_executable NAME SRC FLAMEGPU_ROOT PROJECT_ROOT IS_EXAMPLE)
    # Parse optional arugments.
    cmake_parse_arguments(
        ADD_FLAMEGPU_EXECUTABLE
        ""
        ""
        "LINT_EXCLUDE_FILTERS"
        ${ARGN})

    # If the library does not exist as a target, add it.
    if (NOT TARGET flamegpu2)
        add_subdirectory("${FLAMEGPU_ROOT}/src" "${PROJECT_ROOT}/FLAMEGPU2")
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
    
    # @todo - Once public/private/interface is perfected on the library, some includes may need adding back here.

    # Add extra linker targets
    target_link_libraries(${NAME} ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES})
    
    # Enable RDC for the target
    set_property(TARGET ${NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Link against the flamegpu2 static library target.
    if (TARGET flamegpu2)
        target_link_libraries(${NAME} flamegpu2)
        # Workaround for incremental rebuilds on MSVC, where device link was not being performed.
        if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.1")
            # Provide the absolute path to the lib file, rather than the relative version cmake provides.
            target_link_libraries(${NAME} "${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE:flamegpu2>")
        endif()
    endif()
    
    # Configure device link options
    if(NOT CMAKE_VERSION VERSION_LESS 3.18)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Suppress Fatbinc warnings on msvc at link time.
        target_link_options(${NAME} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler /wd4100>")
        endif()
        if(WARNINGS_AS_ERRORS)               
            if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                target_link_options(${NAME} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler /WX>")
            else()
                target_link_options(${NAME} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler -Werror>")
            endif()
        endif()
    endif()
    
    # Activate visualisation if requested
    if (VISUALISATION)
        target_include_directories(${NAME} PUBLIC "${VISUALISATION_ROOT}/include")
        # Copy DLLs
        if(WIN32)
            # sdl
            set(SDL2_DIR ${VISUALISATION_BUILD}/sdl2)
            mark_as_advanced(FORCE SDL2_DIR)
            find_package(SDL2 REQUIRED)   
            add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD        # Adds a post-build event to MyTest
                COMMAND ${CMAKE_COMMAND} -E copy_if_different             # which executes "cmake - E copy_if_different..."
                    "${SDL2_RUNTIME_LIBRARIES}"                           # <--this is in-file
                    $<TARGET_FILE_DIR:${NAME}>)                           # <--this is out-file path
            # glew
            set(GLEW_DIR ${VISUALISATION_BUILD}/glew)
            mark_as_advanced(FORCE GLEW_DIR)
            find_package(GLEW REQUIRED)   
            add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD        # Adds a post-build event to MyTest
                COMMAND ${CMAKE_COMMAND} -E copy_if_different             # which executes "cmake - E copy_if_different..."
                    "${GLEW_RUNTIME_LIBRARIES}"                           # <--this is in-file
                    $<TARGET_FILE_DIR:${NAME}>)                           # <--this is out-file path
        endif()
        add_compile_definitions(VISUALISATION)
    endif()
    
    # Pass the SEATBELTS macro, which when set to off/0 (for non debug builds) removes expensive operations.
    if (SEATBELTS)
        # If on, all build configs have  seatbelts
        add_compile_definitions(SEATBELTS=1)
    else()
        # Id off, debug builds have seatbelts, non debug builds do not.
        add_compile_definitions($<IF:$<CONFIG:Debug>,SEATBELTS=1,SEATBELTS=0>)
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

# Function to mask some of the flag setting for the static library
function(add_flamegpu_library NAME SRC FLAMEGPU_ROOT)
    # Generate version file
    GET_COMMIT_HASH()
  
    # Define which source files are required for the target executable
    add_library(${NAME} STATIC ${SRC})

    # enable "fpic" for linux to allow shared libraries to be build from the static library (required for swig)
    set_property(TARGET ${NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    # Activate visualisation if requested
    if (VISUALISATION)
        target_include_directories(${NAME} PRIVATE "${VISUALISATION_ROOT}/include")
        target_link_libraries(${NAME} flamegpu2_visualiser)
        CMAKE_SET_TARGET_FOLDER(flamegpu2_visualiser "FLAMEGPU")
        add_compile_definitions(VISUALISATION)
        # set(SDL2_DIR ${VISUALISATION_BUILD}/sdl2)
        # find_package(SDL2 REQUIRED)   
    endif()
    
    # Pass the SEATBELTS macro, which when set to off/0 (for non debug builds) removes expensive operations.
    if (SEATBELTS)
        # If on, all build configs have  seatbelts
        add_compile_definitions(SEATBELTS=1)
    else()
        # Id off, debug builds have seatbelts, non debug builds do not.
        add_compile_definitions($<IF:$<CONFIG:Debug>,SEATBELTS=1,SEATBELTS=0>)
    endif()
    
    if (NOT RTC_DISK_CACHE)
        add_compile_definitions(DISABLE_RTC_DISK_CACHE)
    endif()    
    if (EXPORT_RTC_SOURCES)
        add_compile_definitions(OUTPUT_RTC_DYNAMIC_FILES)
    endif ()
    
    # Enable RDC
    set_property(TARGET ${NAME}  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Link against dependency targets / directories.

    # Linux / not windows has -isystem for suppressing warnings from "system" libraries - ie exeternal dependencies such as thrust.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # CUB (and thrust) cause many compiler warnings at high levels, including Wreorder. 
        # CUB:CUB does not use -isystem to prevent the automatic -I<cuda_path>/include  from being more important, and the CUDA disributed CUB being used. 
        # Instead, if possible we pass the include directory directly rather than using the imported target.
        # And also pass {CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../include" as isystem so the include order is correct for isystem to work (a workaround for a workaround). The `../` is required to prevent cmake from removing the duplicate path.

        # Include CUB via isystem if possible (via _CUB_INCLUDE_DIR which may be subject to change), otherwise use it via target_link_libraries.
        if(DEFINED _CUB_INCLUDE_DIR)
            target_include_directories(${NAME} ${INCLUDE_SYSTEM_FLAG} PUBLIC "${_CUB_INCLUDE_DIR}")
        else()
            target_link_libraries(${NAME} CUB::CUB)
        endif()
        target_include_directories(${NAME}  ${INCLUDE_SYSTEM_FLAG} PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../include")  
    else()
        # MSVC just includes cub via the CUB::CUB target as no isystem to worry about.
        target_link_libraries(${NAME} CUB::CUB)
        # Same for Thrust.
        # Visual studio 2015 needs to suppress deprecation messages from CUB/Thrust.
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
            target_compile_definitions(${NAME} PUBLIC "CUB_IGNORE_DEPRECATED_CPP_DIALECT")
            target_compile_definitions(${NAME} PUBLIC "THRUST_IGNORE_DEPRECATED_CPP_DIALECT")
        endif()
    endif()
        
    # Thrust uses isystem if available
    target_link_libraries(${NAME} Thrust::Thrust)

    # tinyxml2 static library
    target_link_libraries(${NAME} tinyxml2)
    
    # If rapidjson was found, add it to the include dirs.
    if(RapidJSON_FOUND)
        target_include_directories(${NAME} ${INCLUDE_SYSTEM_FLAG} PRIVATE "${RapidJSON_INCLUDE_DIRS}")
    endif()
    
    # Add extra includes (jitify, nvtx, nvrtc etc.) @todo improve this.
    target_include_directories(${NAME}  ${INCLUDE_SYSTEM_FLAG} PUBLIC ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES})

    # Add the library headers as public so they are forwarded on.
    target_include_directories(${NAME}  PUBLIC "${FLAMEGPU_ROOT}/include")
    # Add any private headers.
    target_include_directories(${NAME}  PRIVATE "${FLAMEGPU_ROOT}/src")

    # Add extra linker targets
    target_link_libraries(${NAME} ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES})

    # Flag the new linter target and the files to be linted.
    new_linter_target(${NAME} "${SRC}")
    
    # Put within FLAMEGPU filter
    CMAKE_SET_TARGET_FOLDER(${NAME} "FLAMEGPU")
    # Put the tinyxml2 in the folder
    CMAKE_SET_TARGET_FOLDER("tinyxml2" "FLAMEGPU/Dependencies")

    # Emit some warnings that should only be issued once and are related to this file (but not this target)
    if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL "10.2")
        message(AUTHOR_WARNING "MSVC and NVCC <= 10.2 may encounter compiler errors due to an NVCC bug exposed by Thrust. Cosider using a newer CUDA toolkit.")
    endif()
    if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS_EQUAL "11.0")
        message(AUTHOR_WARNING "MSVC and NVCC <= 11.0 may encounter errors at link time with incremental rebuilds. Cosider using a newer CUDA toolkit.")
    endif()
    if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "11.3")
        message(AUTHOR_WARNING "A workaround for incremental builds is in place for CUDA >= 11.1 which may be unstable in future CUDA versions. See https://github.com:FLAMEGPU/FLAMEGPU2/issues/483")
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

#-----------------------------------------------------------------------
# Generate files that act as informal version numbers
# ${CMAKE_CURRENT_BINARY_DIR}/short_hash.txt - File only contains git short hash for use by CMake
# ${FLAMEGPU_ROOT}/include/flamegpu/version.h - Programatically accessible version
#
# Based on https://cmake.org/pipermail/cmake/2018-October/068388.html
#-----------------------------------------------------------------------
macro(GET_COMMIT_HASH)
# If git changes, we reconfigure
# This is a very aggressive version
# Might be better to simply make generation of the file a pre-build script
# That would be cheaper than re-configure
set_property(
  DIRECTORY 
  APPEND 
  PROPERTY CMAKE_CONFIGURE_DEPENDS 
  "${FLAMEGPU_ROOT}/.git/index"
)
set(SHORT_HASH_FILE ${CMAKE_CURRENT_BINARY_DIR}/short_hash.txt)
find_package(Git)
if(Git_FOUND)
    execute_process(
        COMMAND
            ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY
            ${FLAMEGPU_ROOT}
        RESULT_VARIABLE
            SHORT_HASH_RESULT
        OUTPUT_VARIABLE
            SHORT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    set(SHORT_HASH "GitHash") # Placeholder, though its unlikely to be required
endif()

# Also create version.h
configure_file(${FLAMEGPU_ROOT}/cmake/version.h ${FLAMEGPU_ROOT}/include/flamegpu/version.h)
endmacro()
