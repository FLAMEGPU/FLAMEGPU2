# Set the locations of ARCHIVE (.lib/.a), LIBRARY (MODULE .dll/.so, SHARED.so) and BINARY (.dll,.exe/binary)

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
