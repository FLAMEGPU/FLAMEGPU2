# Define a cmake function which emits a fatal error if the source directory and binary directory are the same.
function(EnforceOutOfSourceBuilds)
    # Resolve paths before comparioson to ensure comparions are accurate
    get_filename_component(source_dir "${CMAKE_SOURCE_DIR}" REALPATH)
    get_filename_component(binary_dir "${CMAKE_BINARY_DIR}" REALPATH)

    if("${source_dir}" STREQUAL "${binary_dir}")
        message(FATAL_ERROR 
            " In-source CMake builds are not allowed.\n"
            " Use a build directory i.e. cmake -B build.\n"
            " You may have to clear/delete the generated CMakeCache.txt and CMakeFiles/:\n"
            "   ${binary_dir}/CMakeCache.txt\n"
            "   ${binary_dir}/CMakeFiles/\n")
    endif()
endfunction()

# Call the function imediately, so the file only needs to be included. 
EnforceOutOfSourceBuilds()

