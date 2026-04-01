#[[[
# Handle CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES gracefully, allowing library CMakeLists.txt to provide a sane library default if not user-specified
#
# CMAKE_CUDA_ARCHITECTURES is a CMake >= 3.18 feature which controls code generation options for CUDA device code.
# The initial value can be set using the CMake Cache variable CMAKE_CUDA_ARCHITECTURES or the CUDAARCHS environment variable.
# 
# CMAKE_HIP_ARCHITECTURES is a CMake >= 3.21 feature which controls code generation options for HIP device code.
# The initial value can be set using the CMake Cache variable CMAKE_HIP_ARCHITECTURES
# The value is interpreted based on CMAKE_HIP_PLATFORM which controls if HIP is compiled for AMD or Nvidia devices.
#
# If CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES is not provided by a users, CMake will default to a (probably) compiler-provide default value in project/enable_language commands, e.g. 75 for CUDA 13.x.
# From this point, we cannot distinguish if the value was user-provided or the CMake default, to override it with a sensible library default (all-major or equivalent).
# This CMake module allows us to set a library default for when users do not provide a specific value, through CMake project injection.
#
# Note: This was formally CUDAArchitectures.cmake, but has been generalised for CUDA/HIP.
#
# Todo: Should we block CMAKE_HIP_PLATFORM=NVIDIA so we don't have to test / support it? 
#
#]]
include_guard(GLOBAL)

#[[[
# Initialise the CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES from the environment, CACHE or a sane programmatic default as appropriate
#
# Call this method prior to the first (or all) project commands, to store the initial state of CMAKE_CUDA_ARCHITECTURES/ENV{CUDAARCHS}/CMAKE_HIP_ARCHITECTURES, enabling custom library-provided default if not specified
# 
# Optionally specify a project name for post-project injection of a call to flamegpu_set_gpu_architectures, to post-process the stored value or set a library-provided default without the user having to provide a value.
#
# :keyword PROJECT: Optional project name to inject CUDA/HIP architecture setting into. Otherwise call flamegpu_set_gpu_architectures manually after the project command or enable_language(CUDA).
# :type PROJECT: string
# :keyword NO_VALIDATE_ARCHITECTURES: Do not validate the passed arguments against nvcc --help output
# :type NO_VALIDATE_ARCHITECTURES: boolean
#]]
function(flamegpu_init_gpu_architectures)
    # Handle argument parsing
    cmake_parse_arguments(FIGA
        "NO_VALIDATE_ARCHITECTURES"
        "PROJECT"
        ""
        ${ARGN}
    )
    # Detect if there are user provided architectures or not, from the cache or environment.
    # This must always be done for all possible GPU architecture sources, as we do not know at this point if HIP or CUDA will be used.
    # Todo: does this need to be split between cuda and hip? how should ENV{CUDAARCHS} be handled for CMAKE_HIP_PLATFORM=nvidia?
    set(flamegpu_ARCH_FROM_ENV_OR_CACHE FALSE)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS} OR DEFINED CMAKE_HIP_ARCHITECTURES)
        set(flamegpu_ARCH_FROM_ENV_OR_CACHE TRUE)
    endif()
    # promote the stored value to parent(file) scope for later use. This might need to become internal cache, but hopefully not.
    set(flamegpu_ARCH_FROM_ENV_OR_CACHE ${flamegpu_ARCH_FROM_ENV_OR_CACHE} PARENT_SCOPE)
    # If the user does not want architecture validation to occur, set a parent scoped variable to be checked later.
    if(FIGA_NO_VALIDATE_ARCHITECTURES)
        # If a project was also specified, append to a list and promote to the parent scope
        if(FIGA_PROJECT)
            list(APPEND flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS ${FIGA_PROJECT})
            set(flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS ${flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS} PARENT_SCOPE)
        else()
            # Otherwise just set a parent scoped variable
            set(flamegpu_NO_VALIDATE_ARCHITECTURES ${FIGA_NO_VALIDATE_ARCHITECTURES} PARENT_SCOPE)
        endif()
    endif()
    # If a project name was provided, inject code into the PROJECT command. Users must call flamegpu_set_gpu_architectures otherwise
    if(FIGA_PROJECT)
        set(CMAKE_PROJECT_${FIGA_PROJECT}_INCLUDE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GPUArchitecturesProjectInclude.cmake" PARENT_SCOPE)
    endif()
endfunction()

#[[[
# Set the CMAKE_CUDA_ARCHITECTURES/CMAKE_HIP_ARCHITECTURES value to the environment/cache provided value, or generate a CUDA/HIP version-appropriate default value for a library (i.e all-major or equivalent)
#
# For CUDA:
#     If the ENV{CUDAARCHS}, or CMAKE_CUDA_ARCHITECTURES cache variable did not specify a value before the CUDA language was enabled, use build an version-appropriate default for a llibrary, based on the CMake and NVCC version.
#     Effectively all-major (-real for all major architectures, and PTX for the most recent)
#
#     If the user provided a value, it will be validated against nvcc --help unless NO_VALIDATE_ARCHITECTURES is set, or was set in a previous call to flamegpu_init_gpu_architectures without a PROJECT.
# 
# For HIP:
#    Todo: figure this out
#    Todo: what is the sensible default for HIP?
#    Todo: what bout hip but with PLATFORM=nvidia? should re-use the cuda bits somehow.
#
# CUDA or HIP must be enabled as a language prior to this method being called.
# Todo: This will be a problem for project-name injection that is also CUDA/HIP compatible. Might have to remove that option...
#
# :keyword NO_VALIDATE_ARCHITECTURES: Do not validate the passed arguments against nvcc --help output
# :type NO_VALIDATE_ARCHITECTURES: boolean
#]]
function(flamegpu_set_gpu_architectures)
    # Handle argument parsing
    cmake_parse_arguments(CSCA
        "NO_VALIDATE_ARCHITECTURES"
        ""
        ""
        ${ARGN}
    )
    # This function requires that the CUDA language is enabled on the current project.
    if(NOT (CMAKE_CUDA_COMPILER_LOADED OR CMAKE_HIP_COMPILER_LOADED))
        # If in the injected project code, give a different error message
        # Todo: is this still viable?
        # Todo: update messages to account for HIP
        if(DEFINED flamegpu_IN_PROJECT_INCLUDE AND flamegpu_IN_PROJECT_INCLUDE)
            message(FATAL_ERROR
            "  ${CMAKE_CURRENT_FUNCTION} requires the CUDA or HIP language to be enabled\n"
            "  Please either:\n"
            # "  *  use project(<project-name> LANGUAGES CUDA)\n"
            "  *  call flamegpu_init_gpu_architectures() without the PROJECT argument, and explicitly call ${CMAKE_CURRENT_FUNCTION}() after flamegpu_enable_languages() / enable_language(CUDA) / enable_language(HIP).")
        else()
            # not in project injection, so only suggest enabled
            message(FATAL_ERROR
            "  ${CMAKE_CURRENT_FUNCTION} requires the CUDA or HIP language to be enabled.\n"
            "  Please call flamegpu_enable_languages(), enable_language(CUDA) or enable_language(HIP) prior to ${CMAKE_CURRENT_FUNCTION}()")
        endif()

    endif()

    # Handle CUDA (Todo: handle hip with nvidia the same way?)
    if (CMAKE_CUDA_COMPILER_LOADED)
        # Query NVCC for the acceptable SM values, this is used in multiple places
        if(NOT DEFINED SUPPORTED_CUDA_ARCHITECTURES_NVCC)
            execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--help" OUTPUT_VARIABLE NVCC_HELP_STR ERROR_VARIABLE NVCC_HELP_STR)
            # Match all comptue_XX or sm_XX or lto_XX values, including single lowercase letter suffixes (a or f only supported so far, but allowing flexibility)
            string(REGEX MATCHALL "'(sm|compute|lto)_([0-9]+[a-z]?)'" SUPPORTED_CUDA_ARCHITECTURES_NVCC "${NVCC_HELP_STR}" )
            # Strip out just the portiaon after the _, which is the value for CMAKE_CUDA_ARCHITECTURES
            string(REGEX REPLACE "'(sm|compute|lto)_([0-9]+[a-z]?)'" "\\2" SUPPORTED_CUDA_ARCHITECTURES_NVCC "${SUPPORTED_CUDA_ARCHITECTURES_NVCC}" )
            # Remove dupes and sort to build the correct list of supported CUDA_ARCH.
            list(REMOVE_DUPLICATES SUPPORTED_CUDA_ARCHITECTURES_NVCC)
            list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES_NVCC "")
            list(SORT SUPPORTED_CUDA_ARCHITECTURES_NVCC)
            # Store the supported arch's once and only once in the parent scope.
            set(SUPPORTED_CUDA_ARCHITECTURES_NVCC ${SUPPORTED_CUDA_ARCHITECTURES_NVCC} PARENT_SCOPE)
        endif()
        list(LENGTH SUPPORTED_CUDA_ARCHITECTURES_NVCC SUPPORTED_CUDA_ARCHITECTURES_NVCC_COUNT)
        # If we already have a cuda architectures value, validate it as CMake doesn't on its own. Unless the caller asked us not to.
        if(flamegpu_ARCH_FROM_ENV_OR_CACHE AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
            AND NOT CSCA_NO_VALIDATE_ARCHITECTURES
            AND NOT flamegpu_NO_VALIDATE_ARCHITECTURES
            AND (DEFINED PROJECT_NAME AND NOT ${PROJECT_NAME} IN_LIST flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS))
            # Get the number or architectures specified
            list(LENGTH CMAKE_CUDA_ARCHITECTURES arch_count)
            # Prep a bool to track if a single special value is being used or not
            set(using_keyword_arch FALSE)
            # native requires CMake >= 3.24, and must be the only option.
            if("native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                # Error if CMake is too old
                if(CMAKE_VERSION VERSION_LESS 3.24)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `native` requires CMake >= 3.24.\n"
                        " CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
                endif()
                # Error if there are multiple architectures specified.
                if(arch_count GREATER 1)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `native` must be the only value specified.\n"
                        " CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
                endif()
                set(using_keyword_arch TRUE)
            endif()
            # all requires 3.23, and must be the sole value.
            if("all" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                # Error if CMake is too old
                if(CMAKE_VERSION VERSION_LESS 3.23)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `all` requires CMake >= 3.23.\n"
                        " CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
                endif()
                # Error if there are multiple architectures specified.
                if(arch_count GREATER 1)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `all` must be the only value specified.\n"
                        " CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
                endif()
                set(using_keyword_arch TRUE)
            endif()
            # all-major requires 3.23, and must be the sole value.
            if("all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES)
                # Error if CMake is too old
                if(CMAKE_VERSION VERSION_LESS 3.23)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `all-major` requires CMake >= 3.23.\n"
                        " CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
                endif()
                # Error if there are multiple architectures specified.
                if(arch_count GREATER 1)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `all-major` must be the only value specified.\n"
                        " CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
                endif()
                set(using_keyword_arch TRUE)
            endif()
            # Cmake 3.18+ expects a list of 1 or more <sm>, <sm>-real or <sm>-virtual.
            # CMake isn't aware of the exact SMS supported by the CUDA version afiak, but we have already queried nvcc for this (once and only once)
            # If nvcc parsing worked and a single keyword option is not being used, attempt the validation
            # But do not emit this more than once per invocation of CMake. parent scope wasn't enough in this case so using a global property
            get_property(
                WARNING_EMITTED
                GLOBAL PROPERTY
                __flamegpu_set_gpu_architectures_warning_emitted
            )
            if(SUPPORTED_CUDA_ARCHITECTURES_NVCC_COUNT GREATER 0 AND NOT using_keyword_arch AND NOT WARNING_EMITTED)
                # Transform a copy of the list of requested architectures, removing -real and -virutal CMake components
                set(archs ${CMAKE_CUDA_ARCHITECTURES})
                list(TRANSFORM archs REPLACE "(\-real|\-virtual)" "")
                # If any of the specified architectures are not in the nvcc reported list, raise a warning
                foreach(ARCH IN LISTS archs)
                    if(NOT ARCH IN_LIST SUPPORTED_CUDA_ARCHITECTURES_NVCC)
                        message(WARNING
                            " CMAKE_CUDA_ARCHITECTURES value `${ARCH}` may not be supported by nvcc ${CMAKE_CUDA_COMPILER_VERSION}, compilation may fail\n"
                            " Supported architectures based on nvcc --help: \n"
                            "   ${SUPPORTED_CUDA_ARCHITECTURES_NVCC}\n")
                    endif()
                endforeach()
                # set the global property so this is not re-emitted for each project()
                set_property(
                    GLOBAL PROPERTY
                    __flamegpu_set_gpu_architectures_warning_emitted
                    TRUE
                )
            endif()
        else()
            # Otherwise, set a mulit-arch default for good compatibility and performacne
            # As we require CMake >= 3.23 (and CUDA >= 12 which includes all-major), we can just use all-major
            set(CMAKE_CUDA_ARCHITECTURES "all-major")
        endif()
        # Promote the value to the parent's scope, where it is needed on the first invokation (might be fine with cache, but just incase)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
        # Promote the value to the cache for reconfigure persistence, as the enable_language sets it on the cache
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
    elseif(CMAKE_HIP_COMPILER_LOADED)
        message(AUTHOR_WARNING "Todo: implement flamegpu_set_gpu_architectures for HIP")
    endif()
endfunction()
