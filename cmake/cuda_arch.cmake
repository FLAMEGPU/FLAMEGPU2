# Build a list of gencode arguments, based on CUDA verison.
# Accepts user override via CUDA_ARCH

# @todo - split setting of values out of the project so it's onyl done once before include_directory?

# Check if any have been provided by the users
string(LENGTH "${CUDA_ARCH}" CUDA_ARCH_LENGTH)

# Define the default compute capabilites incase not provided by the user
set(DEFAULT_CUDA_ARCH "20;35;50;60;70;80;")

# Get the valid options for the current compiler.
# Run nvcc --help to get the help string which contains all valid compute_ sm_ for that version.
execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--help" OUTPUT_VARIABLE NVCC_HELP_STR ERROR_VARIABLE NVCC_HELP_STR)
# Match all comptue_XX or sm_XXs
string(REGEX MATCHALL "'(sm|compute)_[0-9]+'" SUPPORTED_CUDA_ARCH "${NVCC_HELP_STR}" )
# Strip just the numeric component
string(REGEX REPLACE "'(sm|compute)_([0-9]+)'" "\\2" SUPPORTED_CUDA_ARCH "${SUPPORTED_CUDA_ARCH}" )
# Remove dupes and sort to build the correct list of supported CUDA_ARCH.
list(REMOVE_DUPLICATES SUPPORTED_CUDA_ARCH)
list(REMOVE_ITEM SUPPORTED_CUDA_ARCH "")
list(SORT SUPPORTED_CUDA_ARCH)

# Update defaults to only be those supported
# @todo might be better to instead do a dry run compilation with each gencode to validate?
foreach(ARCH IN LISTS DEFAULT_CUDA_ARCH)
    if (NOT ARCH IN_LIST SUPPORTED_CUDA_ARCH)
        list(REMOVE_ITEM DEFAULT_CUDA_ARCH "${ARCH}")
    endif()
    list(REMOVE_DUPLICATES CUDA_ARCH)
    list(REMOVE_ITEM CUDA_ARCH "")
    list(SORT CUDA_ARCH)
endforeach()


if(NOT CUDA_ARCH_LENGTH EQUAL 0)
    # Convert user provided string argument to a list.
    string (REPLACE " " ";" CUDA_ARCH "${CUDA_ARCH}")
    string (REPLACE "," ";" CUDA_ARCH "${CUDA_ARCH}")

    # Remove duplicates, empty items and sort.
    list(REMOVE_DUPLICATES CUDA_ARCH)
    list(REMOVE_ITEM CUDA_ARCH "")
    list(SORT CUDA_ARCH)

    # Validate the list.
    foreach(ARCH IN LISTS CUDA_ARCH)
        if (NOT ARCH IN_LIST SUPPORTED_CUDA_ARCH)
            message(WARNING "Compute Capability ${SM} not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION} and is being ignored.\nChoose from: ${SUPPORTED_CUDA_ARCH}")
            list(REMOVE_ITEM CUDA_ARCH "${ARCH}")
        endif()
    endforeach()

    # @todo - validate that the CUDA_ARCH provided are supported by the compiler
endif()

# If the list is empty post validation, set it to the (validated) defaults
list(LENGTH CUDA_ARCH CUDA_ARCH_LENGTH)
if(CUDA_ARCH_LENGTH EQUAL 0)
    set(CUDA_ARCH ${DEFAULT_CUDA_ARCH})
endif()

# If the list is somehow empty now, do not set any gencodes arguments, instead using the compiler defaults.
list(LENGTH CUDA_ARCH CUDA_ARCH_LENGTH)
if(NOT CUDA_ARCH_LENGTH EQUAL 0)
    message(STATUS "Using Compute Capabilities: ${CUDA_ARCH}")
    set(GENCODES_FLAGS)
    set(MIN_CUDA_ARCH)
    # Convert to gencode arguments

    foreach(ARCH IN LISTS CUDA_ARCH)
        set(GENCODES_FLAGS "${GENCODES_FLAGS} -gencode arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()

    # Add the last arch again as compute_, compute_ to enable forward looking JIT
    list(GET CUDA_ARCH -1 LAST_ARCH)
    set(GENCODES_FLAGS "${GENCODES_FLAGS} -gencode arch=compute_${LAST_ARCH},code=compute_${LAST_ARCH}")

    # Get the minimum device architecture to pass through to nvcc to enable graceful failure prior to cuda execution.
    list(GET CUDA_ARCH 0 MIN_CUDA_ARCH)

    # Set the gencode flags on NVCC
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODES_FLAGS}")

    # Set the minimum arch flags for all compilers
    set(CMAKE_CC_FLAGS "${CMAKE_C_FLAGS} -DMIN_CUDA_ARCH=${MIN_CUDA_ARCH}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIN_CUDA_ARCH=${MIN_CUDA_ARCH}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMIN_CUDA_ARCH=${MIN_CUDA_ARCH}")
else()
    message(STATUS "Using default CUDA Compute Capabilities ${CUDA_ARCH}")
endif()
