# Series of cmake rules that remove absoluate paths from binary (i.e. for reprdoucibility)
# See https://reproducible-builds.org/docs/build-path/ for some informatio non this.

# Define a function which modifies absoluate paths for a given target.
function(TargetStripBuildDirectoryInformation)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        STRIP_BUILD_DIR_INFO
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT STRIP_BUILD_DIR_INFO_TARGET)
        message( FATAL_ERROR "EnableCompilerWarnings: 'TARGET' argument required." )
    elseif(NOT TARGET ${STRIP_BUILD_DIR_INFO_TARGET} )
        message( FATAL_ERROR "EnableCompilerWarnings: TARGET '${STRIP_BUILD_DIR_INFO_TARGET}' is not a valid target" )
    endif()
    # GCC 8+, clang 10+ supports -ffile-prefix-map as an aliase for -fmacro-prefix-map (__FILE__) and -fdebug-prefix-map (Debug info)
    # GCC, clang 3.8+ support -fdebug-prefix-map to strip debug info.
    # NVCC/NVHPC/MSVC do not appear to support similar options.
    if(
        (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8)
        OR 
        (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    )
        target_compile_options(${STRIP_BUILD_DIR_INFO_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>")
        target_compile_options(${STRIP_BUILD_DIR_INFO_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>")
    elseif(
        (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        OR 
        (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "3.8")
    )
        target_compile_options(${STRIP_BUILD_DIR_INFO_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:-fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.>")
        target_compile_options(${STRIP_BUILD_DIR_INFO_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.>")
    endif()
endfunction()