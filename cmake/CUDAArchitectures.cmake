#[[[
# Handle CMAKE_CUDA_ARCHITECTURES gracefully, allowing library CMakeLists.txt to provide a sane default if not user-specified
#
# CMAKE_CUDA_ARCHITECTURES is a CMake >= 3.18 feature which controls code generation options for CUDA device code.
# The initial value can be set using teh CUDAARCHS environment variable, or the CMake Cache varaibel CMAKE_CUDA_ARCHITECTURES.
# This allows users to provide their chosen value.
# When the CUDA language is enabled, by a project() or enable_language() command, if the cache variable is empty then CMake will set the cache variable to the default used by the compiler. I.e. 52 for CUDA 11.x.
# However, it is then impossible to detect if the user provided this value, or if CMake did, preventing a library from setting a default, without executing CMake prior to the first project command, which is unusual.
#
#]]
include_guard(GLOBAL)

#[[[
# Initialise the CMAKE_CUDA_ARCHITECTURES from the environment, CACHE or a sane programatic default
#
# Call this method prior to the first (or all) project commands, to store the initial state of CMAKE_CUDA_ARCHITECTURES/CUDAARCHS for later post processing.
# Optionally specify a project to inject a call to  flamegpu_set_cuda_architectures in to, to post-process the stored value or set a library-provided default.
#
# :keyword PROJECT: Optional project name to inject CMAKE_CUDA_ARCHITECTURES setting into. Otherwise call flamegpu_set_cuda_architectures manually after the project command or enable_lanague(CUDA).
# :type PROJECT: string
# :keyword NO_VALIDATE_ARCHITECTURES: Do not validate the passed arguments against nvcc --help output
# :type NO_VALIDATE_ARCHITECTURES: boolean
#]]
function(flamegpu_init_cuda_architectures)
    # Handle argument parsing
    cmake_parse_arguments(CICA
        "NO_VALIDATE_ARCHITECTURES"
        "PROJECT"
        ""
        ${ARGN}
    )
    # Detect if there are user provided architectures or not, form the cache or environment
    set(flamegpu_ARCH_FROM_ENV_OR_CACHE FALSE)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
        set(flamegpu_ARCH_FROM_ENV_OR_CACHE TRUE)
    endif()
    # promote the stored value to parent(file) scope for later use. This might need to become internal cache, but hopefully not.
    set(flamegpu_ARCH_FROM_ENV_OR_CACHE ${flamegpu_ARCH_FROM_ENV_OR_CACHE} PARENT_SCOPE)
    # If the user does not want architecture validation to occur, set a parent scoped variable to be checked later.
    if(CICA_NO_VALIDATE_ARCHITECTURES)
        # If a project was also specified, append to a list and promote to the parent scope
        if(CICA_PROJECT)
            list(APPEND flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS ${CICA_PROJECT})
            set(flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS ${flamegpu_NO_VALIDATE_ARCHITECTURES_PROJECTS} PARENT_SCOPE)
        else()
            # Otherwise just set a parent scoped variable
            set(flamegpu_NO_VALIDATE_ARCHITECTURES ${CICA_NO_VALIDATE_ARCHITECTURES} PARENT_SCOPE)
        endif()
    endif()
    # If a project name was provided, inject code into the PROJECT command. Users must call flamegpu_set_cuda_architectures otherwise
    if(CICA_PROJECT)
        set(CMAKE_PROJECT_${CICA_PROJECT}_INCLUDE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CUDAArchitecturesProjectInclude.cmake" PARENT_SCOPE)
    endif()
endfunction()

#[[[
# Set the CMAKE_CUDA_ARCHITECTURES value to the environment/cache provided value, or generate the CUDA version appropraite default.
#
# Set teh CMAKE_CUDA_ARCHITECTURES cache variable to a user priovided or a library-arpropriate default.
# If the CUDAARCHS environment variable, or CMAKE_CUDA_ARCHITECTURES cache variable did not specify a value before the CUDA lanagugea was enabled,
# build an appropraite default option, based on the CMake and NVCC verison.
# Effectively all-major (-real for all major achitectures, and PTX for the most reent.)
#
# If the user provided a value, it will be validated against nvcc --help unless NO_VALIDATE_ARCHITECTURES is set, or was set in a previous call to flamegpu_init_cuda_architectures without a PROJECT.
#
# CUDA must be enabled as a language prior to this method being called.
#
# :keyword NO_VALIDATE_ARCHITECTURES: Do not validate the passed arguments against nvcc --help output
# :type NO_VALIDATE_ARCHITECTURES: boolean
#]]
function(flamegpu_set_cuda_architectures)
    # Handle argument parsing
    cmake_parse_arguments(CSCA
        "NO_VALIDATE_ARCHITECTURES"
        ""
        ""
        ${ARGN}
    )
    # This function requires that the CUDA language is enabled on the current project.
    if(NOT CMAKE_CUDA_COMPILER_LOADED)
        # If in the inkected project code, give a different error message
        if(DEFINED flamegpu_IN_PROJECT_INCLUDE AND flamegpu_IN_PROJECT_INCLUDE)
            message(FATAL_ERROR
            "  ${CMAKE_CURRENT_FUNCTION} requires the CUDA lanaguage to be enabled\n"
            "  Please either:\n"
            "  *  use project(<project-name> LANGUAGES CUDA)\n"
            "  *  call flamegpu_init_cuda_architectures() without the PROJECT argument, and explcitly call ${CMAKE_CURRENT_FUNCTION}() after enable_language(CUDA).")
        else()
            # not in project injection, so only suggest enabled
            message(FATAL_ERROR
            "  ${CMAKE_CURRENT_FUNCTION} requires the CUDA language to be enabled.\n"
            "  Please call enable_language(CUDA) prior to ${CMAKE_CURRENT_FUNCTION}()")
        endif()

    endif()
    # Query NVCC for the acceptable SM values, this is used in multiple places
    if(NOT DEFINED SUPPORTED_CUDA_ARCHITECTURES_NVCC)
        execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--help" OUTPUT_VARIABLE NVCC_HELP_STR ERROR_VARIABLE NVCC_HELP_STR)
        # Match all comptue_XX or sm_XXs
        string(REGEX MATCHALL "'(sm|compute)_[0-9]+'" SUPPORTED_CUDA_ARCHITECTURES_NVCC "${NVCC_HELP_STR}" )
        # Strip just the numeric component
        string(REGEX REPLACE "'(sm|compute)_([0-9]+)'" "\\2" SUPPORTED_CUDA_ARCHITECTURES_NVCC "${SUPPORTED_CUDA_ARCHITECTURES_NVCC}" )
        # Remove dupes and sort to build the correct list of supported CUDA_ARCH.
        list(REMOVE_DUPLICATES SUPPORTED_CUDA_ARCHITECTURES_NVCC)
        list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES_NVCC "")
        list(SORT SUPPORTED_CUDA_ARCHITECTURES_NVCC)
        # Store the supported arch's once and only once. This could be a cache  var given the cuda compiler should not be able to change without clearing th cache?
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
        # CMake isn't aware of the exact SMS supported by the CUDA version afiak, but we have already queired nvcc for this (once and only once)
        # If nvcc parsing worked and a single keyword option is not being used, attempt the validation:
        if(SUPPORTED_CUDA_ARCHITECTURES_NVCC_COUNT GREATER 0 AND NOT using_keyword_arch)
            # Transform a copy of the list of supported architectures, to hopefully just contain numbers
            set(archs ${CMAKE_CUDA_ARCHITECTURES})
            list(TRANSFORM archs REPLACE "(\-real|\-virtual)" "")
            # If any of the specified architectures are not in the nvcc reported list, error.
            foreach(ARCH IN LISTS archs)
                if(NOT ARCH IN_LIST SUPPORTED_CUDA_ARCHITECTURES_NVCC)
                    message(FATAL_ERROR
                        " CMAKE_CUDA_ARCHITECTURES value `${ARCH}` is not supported by nvcc ${CMAKE_CUDA_COMPILER_VERSION}.\n"
                        " Supported architectures based on nvcc --help: \n"
                        "   ${SUPPORTED_CUDA_ARCHITECTURES_NVCC}\n")
                endif()
            endforeach()
            unset(archs)
        endif()
    else()
        # Otherwise, set a mulit-arch default for good compatibility and performacne
        # If we're using CMake >= 3.23, we can just use all-major, though we then have to find the minimum a different way?
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
            set(CMAKE_CUDA_ARCHITECTURES "all-major")
        else()
            # For CMake < 3.23, we have to make our own all-major equivalent.
            # If we have nvcc help outut, we can generate this from all the elements that end with a 0 (and the first element if it does not.)
            if(SUPPORTED_CUDA_ARCHITECTURES_NVCC_COUNT GREATER 0)
                # If the lowest support arch is not major, add it to the default
                list(GET SUPPORTED_CUDA_ARCHITECTURES_NVCC 0 lowest_supported)
                if(NOT lowest_supported MATCHES "0$")
                    list(APPEND default_archs ${lowest_supported})
                endif()
                unset(lowest_supported)
                # For each architecture, if it is major add it to the default list
                foreach(ARCH IN LISTS SUPPORTED_CUDA_ARCHITECTURES_NVCC)
                    if(ARCH MATCHES "0$")
                        list(APPEND default_archs ${ARCH})
                    endif()
                endforeach()
            else()
                # If nvcc help output parsing failed, just use an informed guess option from CUDA 12.0
                set(default_archs "35;50;60;70;80")
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
                    list(APPEND default_archs "90")
                endif()
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
                    list(REMOVE_ITEM default_archs "35")
                endif()
                message(AUTHOR_WARNING
                    "  ${CMAKE_CURRENT_FUNCTION} failed to parse NVCC --help output for default architecture generation\n"
                    "  Using ${default_archs} based on CUDA 11.0 to 11.8."
                )
            endif()
            # We actually want real for each arch, then virtual for the final, but only for library-provided values, to only embed one arch worth of ptx.
            # So grab the last element of the list
            list(GET default_archs -1 final)
            # append -real to each element, to not embed ptx for that arch too
            list(TRANSFORM default_archs APPEND "-real")
            # add the -virtual version of the final element
            list(APPEND default_archs "${final}-virtual")
            # Set the value
            set(CMAKE_CUDA_ARCHITECTURES ${default_archs})
            #unset local vars
            unset(default_archs)
        endif()
    endif()
    # Promote the value to the parent's scope, where it is needed on the first invokation (might be fine with cache, but just incase)
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
    # Promote the value to the cache for reconfigure persistence, as the enable_language sets it on the cache
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
endfunction()

#[[[
# Get the minimum CUDA Architecture from the current CMAKE_CUDA_ARCHITECTURES value if possible
#
# Gets the minimum CUDA architectuyre from teh current value of CMAKE_CUDA_ARCHITECTURES if possible, storing the result in the pass-by-reference return value
# Supports CMAKE_CUDA_ARCHITECTURE values including integers, -real post-fixed integers, -virtual post-fixed integers, all-major and all.
# Does not support native, instead returning -1.
# all or all-major are supported by querying nvcc --help to detect the minimum built for.
#
# CUDA must be enabled as a language prior to this method being called, and CMAKE_CUDA_ARCHITECTURES must be defined and non-empty
#
# :param minimum_architecture: the minimum architecture set in CMAKE_CUDA_ARCHITECTURES
# :type NO_VALIDATE_ARCHITECTURES: integer
#]]
function(flamegpu_get_minimum_cuda_architecture minimum_architecture)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Cannot deal with native gracefully
        if("native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
            # If it's native, we would need to exeucte some CUDA code to detect this.
            set(flamegpu_minimum_cuda_architecture 0)
        # if all/all-major is specified, detect via nvcc --help. It must be the only option (CMake doens't validate this and generates bad gencodes otherwise)
        elseif("all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES OR "all" IN_LIST CMAKE_CUDA_ARCHITECTURES)
            # Query NVCC for the acceptable SM values.
            if(NOT DEFINED SUPPORTED_CUDA_ARCHITECTURES_NVCC)
                execute_process(COMMAND ${CMAKE_CUDA_COMPILER} "--help" OUTPUT_VARIABLE NVCC_HELP_STR ERROR_VARIABLE NVCC_HELP_STR)
                # Match all comptue_XX or sm_XXs
                string(REGEX MATCHALL "'(sm|compute)_[0-9]+'" SUPPORTED_CUDA_ARCHITECTURES_NVCC "${NVCC_HELP_STR}" )
                # Strip just the numeric component
                string(REGEX REPLACE "'(sm|compute)_([0-9]+)'" "\\2" SUPPORTED_CUDA_ARCHITECTURES_NVCC "${SUPPORTED_CUDA_ARCHITECTURES_NVCC}" )
                # Remove dupes and sort to build the correct list of supported CUDA_ARCH.
                list(REMOVE_DUPLICATES SUPPORTED_CUDA_ARCHITECTURES_NVCC)
                list(REMOVE_ITEM SUPPORTED_CUDA_ARCHITECTURES_NVCC "")
                list(SORT SUPPORTED_CUDA_ARCHITECTURES_NVCC)
                # Store the supported arch's once and only once. This could be a cache  var given the cuda compiler should not be able to change without clearing th cache?
                set(SUPPORTED_CUDA_ARCHITECTURES_NVCC ${SUPPORTED_CUDA_ARCHITECTURES_NVCC} PARENT_SCOPE)
            endif()
            list(LENGTH SUPPORTED_CUDA_ARCHITECTURES_NVCC SUPPORTED_CUDA_ARCHITECTURES_NVCC_COUNT)
            if(SUPPORTED_CUDA_ARCHITECTURES_NVCC_COUNT GREATER 0)
                # For both all and all-major, the lowest arch should be the lowest supported. This is true for CUDA <= 11.8 atleast.
                list(GET SUPPORTED_CUDA_ARCHITECTURES_NVCC 0 lowest)
                set(flamegpu_minimum_cuda_architecture ${lowest})
            else()
                # If nvcc didn't give anything useful, set to 0
                set(flamegpu_minimum_cuda_architecture 0)
            endif()
        else()
            # Otherwise it should just be a list of one or more <sm>/<sm>-real/<sm-virtual>
            # Copy the list
            set(archs ${CMAKE_CUDA_ARCHITECTURES})
            # Replace occurances of -real and -virtual
            list(TRANSFORM archs REPLACE "(\-real|\-virtual)" "")
            # Sort the list numerically (natural option
            list(SORT archs COMPARE NATURAL ORDER ASCENDING)
            # Get the first element
            list(GET archs 0 lowest)
            # Set the value for later returning
            set(flamegpu_minimum_cuda_architecture ${lowest})
        endif()
        # Set the return value as required, effectively pass by reference.
        set(${minimum_architecture} ${flamegpu_minimum_cuda_architecture} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "${CMAKE_CURRENT_FUNCTION}: CMAKE_CUDA_ARCHITECTURES is not set or is empty")
    endif()
endfunction()
