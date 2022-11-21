# This only gets included once globally, to avoid duplicate warnings (because inclusion includes execution of the method)
include_guard(GLOBAL)
# Define a cmake function which emits a warning if the build directory path contains a space, in some cases
# With Visual Stuido 2022 and CUDA 11.7, this resulted in compilation errors. A relevant bug report has been logged with NVIDIA, so should be fixed in a future CUDA release.
# https://github.com/FLAMEGPU/FLAMEGPU2/issues/867
function(flamegpu_check_binary_dir_for_spaces)
    # If using Visual Studio 17 2022 (the known verison which errors with this, with current CUDA version(s))
    if (CMAKE_GENERATOR MATCHES "Visual Studio 17 2022")
        # Resolve paths to get the full abs path of the binary dir
        get_filename_component(binary_dir "${CMAKE_BINARY_DIR}" REALPATH)

        # Search the binary dir path for the " "
        string(FIND ${binary_dir} " " space_search_result)
        # If the substring was found, emit the warning / error (a value of -1 indicates not found)
        if(space_search_result GREATER_EQUAL 0)
            message(WARNING 
                " The chosen build directory path '${binary_dir}' contains a ' '\n"
                " This is known to cause compilation errors with ${CMAKE_GENERATOR} and CUDA 11.7.\n"
                " Consider using a build directory without a space in the path.\n"
                " Newer CUDA release may have resolved this issue.\n")
        endif()
        unset(space_search_result)
        unset(binary_dir)
    endif()
endfunction()

# Call the function imediately, so the file only needs to be included. 
flamegpu_check_binary_dir_for_spaces()

