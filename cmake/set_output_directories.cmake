# Set CMake variables which influence the location of otuput artifacts such as executables, .so, .dll, .lib files.
# Do this by controlling the default variables, and also providing a function to reset per-target properties to the current global variables.

# Only set these if they have not already been set.
# Set them to be relative to the Project Source directory, i.e. the location of the first call to CMake
#   CMAKE_BINARY_DIR is the top level build directory, so this should achieve the desired effect regardless of whether add_subdirectory is used or not.
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
endif()
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
endif()
if(NOT CMAKE_PDB_OUTPUT_DIRECTORY)
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
endif()

# Provide a target specific method to overwrite the values set on a target to the current values of the global CMAKE variables.
# This is to overwrite per target setting such as in gtest when using add_subdirectory.
function(OverwriteOutputDirectoryProperties)
    # Parse the expected arguments, prefixing variables.
    cmake_parse_arguments(
        OODP
        ""
        "TARGET"
        ""
        ${ARGN}
    )
    # Ensure that a target has been passed, and that it is a valid target.
    if(NOT OODP_TARGET)
        message( FATAL_ERROR "OverwriteOutputDirectoryProperties: 'TARGET' argument required." )
    elseif(NOT TARGET ${OODP_TARGET} )
        message( FATAL_ERROR "OverwriteOutputDirectoryProperties: TARGET '${OVERWRITE_OUTPUT_DIRECTORY_PROPERTIES_TARGET}' is not a valid target" )
    endif()
    # Set the various target properties to current cmake global variable
    set_target_properties(${OODP_TARGET}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        PDB_OUTPUT_DIRECTORY     "${CMAKE_PDB_OUTPUT_DIRECTORY}"
    )
endfunction()