# Function to suppress compiler warnings for a given target
function(DisableCompilerWarnings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        DISABLE_COMPILER_WARNINGS
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT DISABLE_COMPILER_WARNINGS_TARGET)
        message( FATAL_ERROR "DisableCompilerWarnings: 'TARGET' argument required." )
    elseif(NOT TARGET ${DISABLE_COMPILER_WARNINGS_TARGET} )
        message( FATAL_ERROR "DisableCompilerWarnings: TARGET '${DISABLE_COMPILER_WARNINGS_TARGET}' is not a valid target" )
    endif()
    # By default, suppress all warnings, so that warnings are applied per-target.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${DISABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W0>")
    else()
        target_compile_options(${DISABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-w>")
    endif()
    # Always tell nvcc to disable warnings
    target_compile_options(${DISABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-w>")
endfunction()
# Define a function which applies warning flags to a given target.
# Also enables the treating of warnings as errors if required.
function(EnableCompilerWarnings)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        ENABLE_COMPILER_WARNINGS
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT ENABLE_COMPILER_WARNINGS_TARGET)
        message( FATAL_ERROR "EnableCompilerWarnings: 'TARGET' argument required." )
    elseif(NOT TARGET ${ENABLE_COMPILER_WARNINGS_TARGET} )
        message( FATAL_ERROR "EnableCompilerWarnings: TARGET '${ENABLE_COMPILER_WARNINGS_TARGET}' is not a valid target" )
    endif()
    # Host Compiler version specific high warnings
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Only set W4 for MSVC, WAll is more like Wall, Wextra and Wpedantic
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /W4>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>")
        # Also suppress some unwanted W4 warnings
        # decorated name length exceeded, name was truncated
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4503>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4503>")
        # 'function' : unreferenced local function has been removed
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4505>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4505>")
        # unreferenced formal parameter warnings disabled - tests make use of it.
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4100>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4100>")
        # C4127: conditional expression is constant. Longer term true static assertions would be better.
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4127>")
        # Suppress some VS2015 specific warnings.
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4091>")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4091>")
        endif()
    else()
        # Assume using GCC/Clang which Wall is relatively sane for. 
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wall$<COMMA>-Wsign-compare>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall>")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Wsign-compare>")
        # CUB 1.9.10 prevents Wreorder being usable on windows. Cannot suppress via diag_suppress pragmas.
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Wreorder>")
    endif()
    # Promote warnings to errors if requested
    if(WARNINGS_AS_ERRORS)
        # OS Specific flags
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /WX>")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
        else()
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Werror>")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
        endif()
        # Generic WError settings for nvcc
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas=\"-Werror\" -Xnvlink=\"-Werror\">")
        # If CUDA 10.2+, add all_warnings to the Werror option
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "10.2")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>")
        endif()
        # If not msvc, add cross-execution-space-call. This is blocked under msvc by a jitify related bug (untested > CUDA 10.1): https://github.com/NVIDIA/jitify/issues/62
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror cross-execution-space-call >")
        endif()
        # If not msvc, add reorder to Werror. This is blocked under msvc by cub/thrust and the lack of isystem on msvc. Appears unable to suppress the warning via diag_suppress pragmas.
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror reorder >")
        endif()
    endif()
    # Ask the cuda frontend to include warnings numbers, so they can be targetted for suppression.
    target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>")
    # Suppress nodiscard warnings from the cuda frontend
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=2809>")
    endif()
    # Supress CUDA 11.3 specific warnings.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3.0)
        # Suppress 117-D, declared_but_not_referenced
        target_compile_options(${ENABLE_COMPILER_WARNINGS_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=declared_but_not_referenced>")
    endif()
endfunction()