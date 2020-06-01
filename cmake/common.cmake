message(STATUS "-----Configuring Project: ${PROJECT_NAME}-----")
# Add custom modules directory
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})

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
    set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} ${NVRTC_INCLUDE_DIRS})
    set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} ${NVRTC_LIBRARIES})
    # Also add the driver api
    set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} cuda)
else()
    message("nvrtc not found @todo gracefully handle this")
endif()


# If NVTX is enabled, find the library and update variables accordingly.
if(NVTX)
    # Find the nvtx library using custom cmake module
    find_package(NVTX QUIET)
    # If it was found, use it.
    if(NVTX_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNVTX")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNVTX")
        set(FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES} ${NVTX_INCLUDE_DIRS})
        set(FLAMEGPU_DEPENDENCY_LINK_LIBRARIES ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES} ${NVTX_LIBRARIES})
        message("-- Found NVTX: ${NVTX_INCLUDE_DIRS}")
    else()
        # If not found, disable.
        message("NVTX Not found, Setting NVTX=OFF")
        SET(NVTX "OFF")    
    endif()
endif(NVTX)

# Logging for jitify compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DJITIFY_PRINT_LOG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DJITIFY_PRINT_LOG")

# Require a minimum cuda version
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "CUDA version must be at least 7.0")
endif()


# Set Gencode arguments based on cuda version, if not passed in as an argument
# If a list of SMs not passed from the command line, use the defaults
list(LENGTH SMS SMS_COUNT)
if(SMS_COUNT EQUAL 0)
    SET(SMS "")
    # If the CUDA version is less than 8, build for Fermi
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 8.0)
        list(APPEND SMS "20") # Deprecated CUDA 8.0
        list(APPEND SMS "21") # Deprecated CUDA 8.0
    endif()
    # If the CUDA version is >= than 5.0, build for Kepler
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 5.0 )
        # list(APPEND SMS "30") # CUDA >= 5.0  # Skip kepler 1
        list(APPEND SMS "35") # CUDA >= 5.0 
        # list(APPEND SMS "37") # CUDA >= 5.0 # Skip K80s
    endif()
    # If the CUDA version is >= than 5.0, build for Maxwell V1 
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 6.0 )
        list(APPEND SMS "50") # CUDA >= 6.0
    endif()
    # If the CUDA version is >= than 5.0, build for Maxwell V2 
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 7.0 )
        list(APPEND SMS "52") # CUDA >= 6.5
    endif()
    # If the CUDA version is >= 8.0, build for Pascal
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 8.0 )
        list(APPEND SMS "60") # CUDA >= 8.0
        list(APPEND SMS "61") # CUDA >= 8.0
    endif()
    # If the CUDA version is >= 9.0, build for Volta
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 9.0 )
        list(APPEND SMS "70") # CUDA >= 9.0
    endif()
    # If the CUDA version is >= 10.0, build for Turing
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 10.0 )
        list(APPEND SMS "75") # CUDA >= 10.0
    endif()
endif()

# Replace commas and spaces with semicolons to correctly form a cmake list
string (REPLACE " " ";" SMS "${SMS}")
string (REPLACE "," ";" SMS "${SMS}")
SET(SMS "${SMS}" CACHE STRING "compute capabilities to build" FORCE)

# Initialise the variable to contain actual -gencode arguments
SET(GENCODES)
# Remove duplicates from the list of architectures
list(REMOVE_DUPLICATES SMS)
# Remove empty items from the list of architectures
list(REMOVE_ITEM SMS "")
# Sort the list of SM architectures into ascending order.
list(SORT SMS)
# For each SM, generate the relevant -gencode argument
foreach(SM IN LISTS SMS)
    set(GENCODES "${GENCODES} -gencode arch=compute_${SM},code=sm_${SM}")
endforeach()

# Get the minimum device architecture to pass through to nvcc to enable graceful failure prior to cuda execution.
list(GET SMS 0 MIN_ARCH)
# Pass this to the compiler(s)
SET(CMAKE_CC_FLAGS "${CMAKE_C_FLAGS} -DMIN_ARCH=${MIN_ARCH}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIN_ARCH=${MIN_ARCH}")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMIN_ARCH=${MIN_ARCH}")

# Using the last element of the list, append the additional gencode argument
list(GET SMS -1 LAST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LAST_SM},code=compute_${LAST_SM}")

# Append the gencodes to the nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODES}")

# Don't create this message multiple times
if(NOT COMMAND add_flamegpu_executable)
    # Output the GENCODES to the user.
    message(STATUS "Targeting Compute Capabilities: ${SMS}")
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
# Set high level of warnings (only for linux due to Jitify bug: https://github.com/NVIDIA/jitify/issues/62)
if(WARNINGS_AS_ERRORS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Jitify has a problem with cross-execution-space-call under windows, enabling that currently blocks appveyor
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wreorder --Werror reorder -Xptxas=\"-Werror\"  -Xnvlink=\"-Werror\"")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wreorder --Werror reorder,cross-execution-space-call -Xptxas=\"-Werror\"  -Xnvlink=\"-Werror\"")
    endif()
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wreorder")
endif()
# Compiler version specific high warnings
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
    set(CUDA_DEVICE_LINK_FLAGS "${CUDA_DEVICE_LINK_FLAGS} -Xcompiler /wd4100")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /experimental:external")
    # These flags don't currently have any effect on how CMake passes system-private includes to msvc
    set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "/external:I")
    set(CMAKE_INCLUDE_SYSTEM_FLAG_CUDA "/external:I")
    if(WARNINGS_AS_ERRORS)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /WX")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
        set(CUDA_DEVICE_LINK_FLAGS "${CUDA_DEVICE_LINK_FLAGS} -Xcompiler /WX")
    endif()
else()
    # Assume using GCC/Clang which Wall is relatively sane for. 
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wsign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wsign-compare")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wsign-compare")
    if(WARNINGS_AS_ERRORS)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Werror")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
        set(CUDA_DEVICE_LINK_FLAGS "${CUDA_DEVICE_LINK_FLAGS} -Xcompiler -Werror")
    endif()
endif()
# Common CUDA args

# Use C++14 standard - std::make_unique is 14 not 11
# Specify using C++14 standard
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()

# Tell CUDA to use C++14 standard
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
    function(new_linter_target NAME SRC)
        # Don't lint external files
        list(FILTER SRC EXCLUDE REGEX "^${FLAMEGPU_ROOT}/externals/.*")
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

    # If the library does not exist as a target, add it.
    if (NOT TARGET flamegpu2)
        add_subdirectory("${FLAMEGPU_ROOT}/src" "${PROJECT_ROOT}/FLAMEGPU2")
    endif()

    # Define which source files are required for the target executable
    add_executable(${NAME} ${SRC})
    
    # Add include directories
    target_include_directories(${NAME} ${INCLUDE_SYSTEM_FLAG} PRIVATE ${FLAMEGPU_ROOT}/externals)
    # Add the cuda include directory as a system include to allow user-provided thrust. ../include trickery for cmake >= 3.12
    target_include_directories(${NAME} ${INCLUDE_SYSTEM_FLAG} PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../include")    
    target_include_directories(${NAME} ${INCLUDE_SYSTEM_FLAG} PRIVATE ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES})
    target_include_directories(${NAME} PRIVATE ${FLAMEGPU_ROOT}/include)

    # Add extra linker targets
    target_link_libraries(${NAME} ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES})
    
    # Enable RDC for the target
    set_property(TARGET ${NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Link against the flamegpu2 static library target.
    if (TARGET flamegpu2)
        target_link_libraries(${NAME} flamegpu2)
    endif()
    
    # Activate visualisation if requested
    if (VISUALISATION)
        target_include_directories(${NAME} PRIVATE ${VISUALISATION_ROOT}/include)
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
        # fgpu2 lib will add this dependency
        # target_link_libraries(${NAME} flamegpu2_visualiser)
        add_compile_definitions(VISUALISATION)
    endif()

    # Flag the new linter target and the files to be linted.
    new_linter_target(${NAME} "${SRC}")
    
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
    list(FILTER T_SRC EXCLUDE REGEX ".*\.(h|hpp|cuh)$")
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX src FILES ${T_SRC})


    # Put within Examples filter
    if(IS_EXAMPLE)
        CMAKE_SET_TARGET_FOLDER(${NAME} "Examples")
    endif()
endfunction()

# Function to mask some of the flag setting for the static library
function(add_flamegpu_library NAME SRC FLAMEGPU_ROOT)
    # Define which source files are required for the target executable
    add_library(${NAME} STATIC ${SRC})
    
    # Activate visualisation if requested
    if (VISUALISATION)
        target_include_directories(${NAME} PRIVATE ${VISUALISATION_ROOT}/include)
        target_link_libraries(${NAME} flamegpu2_visualiser)
        CMAKE_SET_TARGET_FOLDER(flamegpu2_visualiser "FLAMEGPU")
        add_compile_definitions(VISUALISATION)
        # set(SDL2_DIR ${VISUALISATION_BUILD}/sdl2)
        # find_package(SDL2 REQUIRED)   
    endif()

    # Enable RDC
    set_property(TARGET ${NAME}  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Define include dirs
    target_include_directories(${NAME} ${INCLUDE_SYSTEM_FLAG} PRIVATE ${FLAMEGPU_ROOT}/externals)
    # Add the cuda include directory as a system include to allow user-provided thrust. ../include trickerty for cmake >= 3.12
    target_include_directories(${NAME}  ${INCLUDE_SYSTEM_FLAG} PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../include")
    target_include_directories(${NAME}  ${INCLUDE_SYSTEM_FLAG} PRIVATE ${FLAMEGPU_DEPENDENCY_INCLUDE_DIRECTORIES})
    target_include_directories(${NAME}  PRIVATE ${FLAMEGPU_ROOT}/include)
    target_include_directories(${NAME}  PRIVATE ${FLAMEGPU_ROOT}/src) #private headers

    # Add extra linker targets
    target_link_libraries(${NAME} ${FLAMEGPU_DEPENDENCY_LINK_LIBRARIES})

    # Flag the new linter target and the files to be linted.
    new_linter_target(${NAME} "${SRC}")
    
    # Put within FLAMEGPU filter
    CMAKE_SET_TARGET_FOLDER(${NAME} "FLAMEGPU")
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
