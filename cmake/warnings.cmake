# Function to disable all (as many as possible) compiler warnings for a given target
function(DisableCompilerWarnings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        DCW
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT DCW_TARGET)
        message(FATAL_ERROR "DisableCompilerWarnings: 'TARGET' argument required.")
    elseif(NOT TARGET ${DCW_TARGET})
        message(FATAL_ERROR "DisableCompilerWarnings: TARGET '${DCW_TARGET}' is not a valid target")
    endif()
    # By default, suppress all warnings, so that warnings are applied per-target.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${DCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W0>")
    else()
        target_compile_options(${DCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-w>")
    endif()
    # Always tell nvcc to disable warnings
    target_compile_options(${DCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-w>")
endfunction()

# Function to set a high level of compiler warnings for a target
# Function to disable all (as many as possible) compiler warnings for a given target
function(SetHighWarningLevel)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        SHWL
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT SHWL_TARGET)
        message(FATAL_ERROR "SetHighWarningLevel: 'TARGET' argument required.")
    elseif(NOT TARGET ${SHWL_TARGET})
        message(FATAL_ERROR "SetHighWarningLevel: TARGET '${SHWL_TARGET}' is not a valid target")
    endif()
    
    # Per host-compiler settings for high warning levels and opt-in warnings.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Only set W4 for MSVC, WAll is more like Wall, Wextra and Wpedantic
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /W4>")
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>")
    else()
        # Assume using GCC/Clang which Wall is relatively sane for. 
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wall$<COMMA>-Wsign-compare>")
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall>")
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wsign-compare>")
        # CUB 1.9.10 prevents Wreorder being usable on windows, so linux only. Cannot suppress via diag_suppress pragmas.
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wreorder>")
        # Add warnings which suggest the use of override
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wsuggest-override>")
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wsuggest-override>")
    endif()
    # Generic options regardless of platform/host compiler:
    # Ensure NVCC outputs warning numbers
    target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>")
endfunction()

# Function to apply warning suppressions to a given target, without changing the general warning level (This is so SWIG can have suppressions, with default warning levels)
function(SuppressSomeCompilerWarnings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        SSCW
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT SSCW_TARGET)
        message(FATAL_ERROR "SuppressSomeCompilerWarnings: 'TARGET' argument required.")
    elseif(NOT TARGET ${SSCW_TARGET})
        message(FATAL_ERROR "SuppressSomeCompilerWarnings: TARGET '${SSCW_TARGET}' is not a valid target")
    endif()

    # Per host-compiler/OS settings for suppressions
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # decorated name length exceeded, name was truncated
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4503>")
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4503>")
        # 'function' : unreferenced local function has been removed
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4505>")
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4505>")
        # unreferenced formal parameter warnings disabled - tests make use of it.
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4100>")
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4100>")
        # C4127: conditional expression is constant. Longer term true static assertions would be better.
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4127>")
        # Suppress some VS2015 specific warnings.
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4091>")
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4091>")
        endif()
        # Suppress Fatbinc warnings on msvc at link time (CMake >= 3.18)
        target_link_options(${SSCW_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler /wd4100>")
    else()
        # Linux specific warning suppressions
    endif()
    # Generic OS/host compiler warning suppressions
    # Ensure NVCC outputs warning numbers
    target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>")
    # Suppress deprecated compute capability warnings.
    target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")
    target_link_options(${SSCW_TARGET} PRIVATE "$<DEVICE_LINK:-Wno-deprecated-gpu-targets>")
    # Supress CUDA 11.3 specific warnings, which are host compiler agnostic.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3.0)
        # Suppress 117-D, declared_but_not_referenced
        target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=declared_but_not_referenced>")
    endif()
    # Suppress nodiscard warnings from the cuda frontend
    target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=2809>")
endfunction()

# Function to promote warnings to errors, controlled by the WARNINGS_AS_ERRORS CMake option.
function(EnableWarningsAsErrors)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        EWAS
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT EWAS_TARGET)
        message(FATAL_ERROR "EnableWarningsAsErrors: 'TARGET' argument required.")
    elseif(NOT TARGET ${EWAS_TARGET})
        message(FATAL_ERROR "EnableWarningsAsErrors: TARGET '${EWAS_TARGET}' is not a valid target")
    endif()
    
    # Check the WARNINGS_AS_ERRORS cmake option to optionally enable this.
    if(WARNINGS_AS_ERRORS)
        # OS Specific flags
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # Windows specific options
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /WX>")
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
            # Device link warnings as errors, CMake 3.18+
            target_link_options(${EWAS_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler /WX>")
        else()
            # Linux specific options
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Werror>")
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
            # Device link warnings as errors, CMake 3.18+
            target_link_options(${EWAS_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler -Werror>")
            # Add cross-execution-space-call. This is blocked under msvc by a jitify related bug (untested > CUDA 10.1): https://github.com/NVIDIA/jitify/issues/62
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror cross-execution-space-call>")
            # Add reorder to Werror. This is blocked under msvc by cub/thrust and the lack of isystem on msvc. Appears unable to suppress the warning via diag_suppress pragmas.
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror reorder>")
        endif()
        # Platform/host-compiler indifferent options:
        # Generic WError settings for nvcc
        target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas=\"-Werror\" -Xnvlink=\"-Werror\">")
        # If CUDA 10.2+, add all_warnings to the Werror option
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.2")
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>")
        endif()
    endif()
endfunction()


# Define a function which sets our preffered warning options for a target:
# + A high warning level
# + With some warnings suppressed
# + Optionally promotes warnings to errors.
# Also enables the treating of warnings as errors if required.
function(EnableFLAMEGPUCompilerWarnings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        EFCW
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT EFCW_TARGET)
        message(FATAL_ERROR "EnableFLAMEGPUCompilerWarnings: 'TARGET' argument required.")
    elseif(NOT TARGET ${EFCW_TARGET})
        message(FATAL_ERROR "EnableFLAMEGPUCompilerWarnings: TARGET '${EFCW_TARGET}' is not a valid target")
    endif()

    # Enable a high level of warnings
    SetHighWarningLevel(TARGET ${EFCW_TARGET})
    # Suppress some warnings
    SuppressSomeCompilerWarnings(TARGET ${EFCW_TARGET})
    # Optionally promote warnings to errors.
    EnableWarningsAsErrors(TARGET ${EFCW_TARGET})
endfunction()