# Provides a per target function to set gencode compiler options.
# Function to suppress compiler warnings for a given target
# If the cmake variable CUDA_ARCH is set, to a non emtpy list or space separated string this will be used instead.
# @todo - find a way to warn about  deprecated architectures once and only once (at cmake time?) Might need to just try compiling with old warnings and capture / post process the output. 
# @todo - figure out how to do this once and only once as a function rather than a macro.
macro(SetCUDAGencodes)
    # @todo - only get the available gencodes from nvcc once, rather than per target.

    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        SCG
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT SCG_TARGET)
        message( FATAL_ERROR "SetCUDAGencodes: 'TARGET' argument required." )
    elseif(NOT TARGET ${SCG_TARGET} )
        message( FATAL_ERROR "SetCUDAGencodes: TARGET '${SCG_TARGET}' is not a valid target" )
    endif()

    # CMAKE > 3.18 introduces CUDA_ARCHITECTURES as a cmake-native way of generating gencodes (Policy CMP0104). Set the value to OFF to prevent errors for it being not provided.
    # We manually set gencode arguments, so we can (potentially) use LTO and are not restricted to CMake's availble options.
    set_property(TARGET ${SCG_TARGET} PROPERTY CUDA_ARCHITECTURES OFF)

    # Define the default compute capabilites incase not provided by the user
    set(DEFAULT_CUDA_ARCH "35;50;60;70;80;90;")

    # Determine if the user has provided a non default CUDA_ARCH value 
    string(LENGTH "${CUDA_ARCH}" CUDA_ARCH_LENGTH)

    # Query NVCC in order to filter the provided list. 
    # @todo only do this once, and re-use the output for a given cmake configure?

    # Get the valid options for the current compiler.
    # Run nvcc --help to get the help string which contains all valid compute_ sm_ for that version.
    if(NOT DEFINED SUPPORTED_CUDA_ARCH)
        execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--help" OUTPUT_VARIABLE NVCC_HELP_STR ERROR_VARIABLE NVCC_HELP_STR)
        # Match all comptue_XX or sm_XXs
        string(REGEX MATCHALL "'(sm|compute)_[0-9]+'" SUPPORTED_CUDA_ARCH "${NVCC_HELP_STR}" )
        # Strip just the numeric component
        string(REGEX REPLACE "'(sm|compute)_([0-9]+)'" "\\2" SUPPORTED_CUDA_ARCH "${SUPPORTED_CUDA_ARCH}" )
        # Remove dupes and sort to build the correct list of supported CUDA_ARCH.
        list(REMOVE_DUPLICATES SUPPORTED_CUDA_ARCH)
        list(REMOVE_ITEM SUPPORTED_CUDA_ARCH "")
        list(SORT SUPPORTED_CUDA_ARCH)

        # Store the supported arch's once and only once. This could be a cache  var given the cuda compiler should not be able to change without clearing th cache?
        get_directory_property(hasParent PARENT_DIRECTORY)
        if(hasParent)
            set(SUPPORTED_CUDA_ARCH ${SUPPORTED_CUDA_ARCH} PARENT_SCOPE)
        endif()
    endif()
    
    
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
                message(WARNING
                "  CUDA_ARCH '${ARCH}' not supported by CUDA ${CMAKE_CUDA_COMPILER_VERSION} and is being ignored.\n"
                "  Choose from: ${SUPPORTED_CUDA_ARCH}")
                list(REMOVE_ITEM CUDA_ARCH "${ARCH}")
            endif()
        endforeach()
    endif()


    # If the list is empty post validation, set it to the (validated) defaults
    list(LENGTH CUDA_ARCH CUDA_ARCH_LENGTH)
    if(CUDA_ARCH_LENGTH EQUAL 0)
        set(CUDA_ARCH ${DEFAULT_CUDA_ARCH})
    endif()

    # Propagate the validated values to the parent scope, to reduce warning duplication.
    get_directory_property(hasParent PARENT_DIRECTORY)
    if(hasParent)
        set(CUDA_ARCH ${CUDA_ARCH} PARENT_SCOPE)
    endif()

    # If the list is somehow empty now, do not set any gencodes arguments, instead using the compiler defaults.
    list(LENGTH CUDA_ARCH CUDA_ARCH_LENGTH)
    if(NOT CUDA_ARCH_LENGTH EQUAL 0)
        # Only do this if required.I.e. CUDA_ARCH is the same as the last time this file was included
        if(NOT CUDA_ARCH_APPLIED EQUAL CUDA_ARCH)
            message(STATUS "Generating Compute Capabilities: ${CUDA_ARCH}")
            if(hasParent)
                set(CUDA_ARCH_APPLIED "${CUDA_ARCH}" PARENT_SCOPE )
            endif()
        endif()
        set(MIN_CUDA_ARCH)
        # Convert to gencode arguments

        foreach(ARCH IN LISTS CUDA_ARCH)
            target_compile_options(${SCG_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_${ARCH}$<COMMA>code=sm_${ARCH}>")
            target_link_options(${SCG_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-gencode arch=compute_${ARCH}$<COMMA>code=sm_${ARCH}>")
        endforeach()

        # Add the last arch again as compute_, compute_ to enable forward looking JIT
        list(GET CUDA_ARCH -1 LAST_ARCH)
        target_compile_options(${SCG_TARGET} PRIVATE  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_${LAST_ARCH}$<COMMA>code=compute_${LAST_ARCH}>")
        target_link_options(${SCG_TARGET} PRIVATE  "$<DEVICE_LINK:SHELL:-gencode arch=compute_${LAST_ARCH}$<COMMA>code=compute_${LAST_ARCH}>")

        # Get the minimum device architecture to pass through to nvcc to enable graceful failure prior to cuda execution.
        list(GET CUDA_ARCH 0 MIN_CUDA_ARCH)

        # Set the minimum arch flags for all compilers
        target_compile_definitions(${SCG_TARGET} PRIVATE -DMIN_CUDA_ARCH=${MIN_CUDA_ARCH})
    else()
        message(STATUS "Generating default CUDA Compute Capabilities ${CUDA_ARCH}")
    endif()
endmacro()
