message(STATUS "-----Configuring Project: ${PROJECT_NAME}-----")
# Common rules for other cmake files
STRING(TOLOWER "${CMAKE_SYSTEM_NAME}" CMAKE_SYSTEM_NAME_LOWER)

# Set a default build type if not passed (https://blog.kitware.com/cmake-and-the-default-build-type/)
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "Profile")
endif()

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES Debug Release Profile)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
    "Reset the configurations to what we need"
    FORCE)
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
# Update the documentation string of CMAKE_BUILD_TYPE for GUIs
SET( CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Choose the type of build, options are: None Debug Release Profile."
    FORCE )




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
        list(APPEND SMS "37") # CUDA >= 5.0 
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

# Replace commas and spaces with semicolons to correclty form a cmake list
string (REPLACE " " ";" SMS "${SMS}")
string (REPLACE "," ";" SMS "${SMS}")

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

# Using the last element of the list, append the additional gencode argument
list(GET SMS -1 LAST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LAST_SM},code=compute_${LAST_SM}")

# Append the gencodes to the nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODES}")

# Output the GENCODES to teh user.
message(STATUS "Targeting Compute Capabilities: ${SMS}")

# Specify some additional compiler flags
# CUDA debug symbols
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -D_DEBUG -DDEBUG")

# Lineinfo for non -G release
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

# profile specific CUDA flags.
set(CMAKE_CUDA_FLAGS_PROFILE "${CMAKE_CUDA_FLAGS_PROFILE} -lineinfo -DPROFILE -D_PROFILE")

# Set high level of warnings, specific to the host compiler.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Only set W4 for MSVC, WAll is more like Wall, Wextra and Wpedantic
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    # Also suppress some unwanted W4 warnings
    # 'function' : unreferenced local function has been removed
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /wd4505")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4505")
else()
    # Assume using GCC/Clang which Wall is relatively sane for. 
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

# Use C++14 standard - std::make_unique is 14 not 11
# Specify using C++14 standard
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()

# Tell CUDA to use C++14 standard
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if(CPPLINT)
    function(new_linter_target NAME SRC)
        # Add custom target for linting this
        add_custom_target(
            "lint_${NAME}"
            COMMAND cpplint
            ${SRC}
        )
        # Add the custom target as a dependency of the global lint target
        if(TARGET lint)
            add_dependencies(lint lint_${NAME})
        endif()
    endfunction()
else()
    message(WARNING "cpplint script: NOT FOUND! "
                    "Lint projects will not be generated"
                    "Please install cpplint as described on https://pypi.python.org/pypi/cpplint. "
			        "In most cases command '(sudo )pip install cpplint' should be sufficient.")
    function(new_linter_target NAME SRC)
    endfunction()
endif()

# Function to mask some of the steps to create an executable which links against the static library
function(add_flamegpu_executable NAME SRC FLAMEGPU_ROOT)

    # If the library does not exist as a target, add it.
    if (NOT TARGET flamegpu2)
        # Use the expected build directory location for the static library, so that multiple examples can be built without the need for a rebuild.
        add_subdirectory(${FLAMEGPU_ROOT}/src ${FLAMEGPU_ROOT}/src/build) # Use the src/build dir to avoid re-builds
        # add_subdirectory(${FLAMEGPU_ROOT}/src ./flamegpu2) # Use a child directory for more locality
    endif()

    # Define which source files are required for the target executable
    add_executable(${NAME} ${SRC})

    # Add include directories
    target_include_directories(${NAME} PRIVATE ${FLAMEGPU_ROOT}/include)

    # Enable RDC for the target
    set_property(TARGET ${NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Link against the flamegpu2 static library target.
    if (TARGET flamegpu2)
        target_link_libraries(${NAME} flamegpu2)
    endif()

    # Flag the new linter target and the files to be linted.
    new_linter_target(${NAME} ${SRC})


endfunction()

# Function to mask some of the flag setting for the static library
function(add_flamegpu_library NAME SRC FLAMEGPU_ROOT)
    # Define which source files are required for the target executable
    add_library(${NAME} STATIC ${SRC})

    # Enable RDC
    set_property(TARGET ${NAME}  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Define include dirs
    target_include_directories(${NAME}  PRIVATE ${FLAMEGPU_ROOT}/include)
    target_include_directories(${NAME}  PRIVATE ${FLAMEGPU_ROOT}/externals)

    # Flag the new linter target and the files to be linted.
    new_linter_target(${NAME} ${SRC})

endfunction()
