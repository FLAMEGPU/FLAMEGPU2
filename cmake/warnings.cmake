include_guard(GLOBAL)

# Function to disable all (as many as possible) compiler warnings for a given target
if(NOT COMMAND flamegpu_disable_compiler_warnings)
    function(flamegpu_disable_compiler_warnings)
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
            message(FATAL_ERROR "flamegpu_disable_compiler_warnings: 'TARGET' argument required.")
        elseif(NOT TARGET ${DCW_TARGET})
            message(FATAL_ERROR "flamegpu_disable_compiler_warnings: TARGET '${DCW_TARGET}' is not a valid target")
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
endif()

# Function to set a high level of compiler warnings for a target
# Function to disable all (as many as possible) compiler warnings for a given target
if(NOT COMMAND flamegpu_set_high_warning_level)
    function(flamegpu_set_high_warning_level)
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
            message(FATAL_ERROR "flamegpu_set_high_warning_level: 'TARGET' argument required.")
        elseif(NOT TARGET ${SHWL_TARGET})
            message(FATAL_ERROR "flamegpu_set_high_warning_level: TARGET '${SHWL_TARGET}' is not a valid target")
        endif()
        
        # Per host-compiler settings for high warning levels and opt-in warnings.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # Only set W4 for MSVC, WAll is more like Wall, Wextra and Wpedantic
            target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /W4>")
            target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>")
            # Reorder errors for device code are caused by some cub/thrust versions (< 2.1.0?), but can be suppressed by pragmas successfully in 11.5+ under windows
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5.0)
                target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wreorder>")
            endif()
        else()
            # Assume using GCC/Clang which Wall is relatively sane for. 
            target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wall$<COMMA>-Wsign-compare>")
            target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall>")
            target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wsign-compare>")
            # Reorder errors for device code are caused by some cub/thrust versions (< 2.1.0?), but can be suppressed by pragmas successfully in 11.3+ under linux
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3.0)
                target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wreorder>")
            endif()
            # Add warnings which suggest the use of override
            # Disabled, as cpplint occasionally disagrees with gcc concerning override
            # target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wsuggest-override>")
            # target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wsuggest-override>")
        endif()
        # Generic options regardless of platform/host compiler:
        # Ensure NVCC outputs warning numbers
        target_compile_options(${SHWL_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>")
    endfunction()
endif()

# Function to apply warning suppressions to a given target, without changing the general warning level (This is so SWIG can have suppressions, with default warning levels)
if(NOT COMMAND flamegpu_suppress_some_compiler_warnings)
    function(flamegpu_suppress_some_compiler_warnings)
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
            message(FATAL_ERROR "flamegpu_suppress_some_compiler_warnings: 'TARGET' argument required.")
        elseif(NOT TARGET ${SSCW_TARGET})
            message(FATAL_ERROR "flamegpu_suppress_some_compiler_warnings: TARGET '${SSCW_TARGET}' is not a valid target")
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
            # Suppress Fatbinc warnings on msvc at link time (CMake >= 3.18)
            target_link_options(${SSCW_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler /wd4100>")
            # CUDA 11.6 deprecates __device__ cudaDeviceSynchronize, but does not provide an alternative.
            # This is used in cub/thrust, and windows still emits this warning from the third party library
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.6.0)
                target_compile_definitions(${SSCW_TARGET} PRIVATE "__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING")
            endif()
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # Suppress unused function warnigns raised by clang on some vis headers
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-unused-function>")
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-function>")
            # Suppress unused-private-field warnings on Clang, which are falsely emitted in some cases where a private member is used in device code (i.e. ArrayMessage)
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-unused-private-field>")
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-private-field>")
            # Suppress unused-but-set-variable which triggers on some device code, clang 13+
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
                target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-unused-but-set-variable>")
                target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-but-set-variable>")
            endif()
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
        # Suppress Power9 + GCC >= 10 note re: ABI changes in GCC >= 5
        # "Note: the layout of aggregates containing vectors with x-byte allignment has changed in GCC 5
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
            target_compile_options(${SSCW_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:>-Wno-psabi")
        endif()
    endfunction()
endif()

# Function to promote warnings to errors, controlled by the FLAMEGPU_WARNINGS_AS_ERRORS CMake option.
if(NOT COMMAND flamegpu_enable_warnings_as_errors)
    function(flamegpu_enable_warnings_as_errors)
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
            message(FATAL_ERROR "flamegpu_enable_warnings_as_errors: 'TARGET' argument required.")
        elseif(NOT TARGET ${EWAS_TARGET})
            message(FATAL_ERROR "flamegpu_enable_warnings_as_errors: TARGET '${EWAS_TARGET}' is not a valid target")
        endif()
        
        # Check the FLAMEGPU_WARNINGS_AS_ERRORS cmake option to optionally enable this.
        if(FLAMEGPU_WARNINGS_AS_ERRORS)
            # OS Specific flags
            if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                # Windows specific options
                target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /WX>")
                target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
                # Device link warnings as errors, CMake 3.18+
                target_link_options(${EWAS_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler /WX>")
                # Add reorder to Werror, this is usable with workign nv/diag_suppress pragmas for cub/thrust from CUDA 11.5+ under windows
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5.0)
                    target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror reorder>")
                endif()
            else()
                # Linux specific options
                target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Werror>")
                target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
                # Device link warnings as errors, CMake 3.18+
                target_link_options(${EWAS_TARGET} PRIVATE "$<DEVICE_LINK:SHELL:-Xcompiler -Werror>")
                # Add cross-execution-space-call. This is blocked under msvc by a jitify related bug (untested > CUDA 10.1): https://github.com/NVIDIA/jitify/issues/62
                target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror cross-execution-space-call>")
                # Add reorder to Werror, this is usable with workign nv/diag_suppress pragmas for cub/thrust from CUDA 11.3+ under linux
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3.0)
                    target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror reorder>")
                endif()
            endif()
            # Platform/host-compiler indifferent options:
            # Generic WError settings for nvcc
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas=\"-Werror\" -Xnvlink=\"-Werror\">")
            # Add all_warnings to the Werror option (supported by all CUDA 11.x+)
            target_compile_options(${EWAS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>")
        endif()
    endfunction()
endif()

# Define a function which sets our preffered warning options for a target:
# + A high warning level
# + With some warnings suppressed
# + Optionally promotes warnings to errors.
# Also enables the treating of warnings as errors if required.
if(NOT COMMAND flamegpu_enable_compiler_warnings)
    function(flamegpu_enable_compiler_warnings)
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
            message(FATAL_ERROR "flamegpu_enable_compiler_warnings: 'TARGET' argument required.")
        elseif(NOT TARGET ${EFCW_TARGET})
            message(FATAL_ERROR "flamegpu_enable_compiler_warnings: TARGET '${EFCW_TARGET}' is not a valid target")
        endif()

        # Enable a high level of warnings
        flamegpu_set_high_warning_level(TARGET ${EFCW_TARGET})
        # Suppress some warnings
        flamegpu_suppress_some_compiler_warnings(TARGET ${EFCW_TARGET})
        # Optionally promote warnings to errors.
        flamegpu_enable_warnings_as_errors(TARGET ${EFCW_TARGET})
    endfunction()
endif()
