include_guard(GLOBAL)

# Define a cmake function which checks that CUDA and the host compiler are functional.
# Failure only results in CMake warnings not Errors, so that documentation only builds will function.
function(flamegpu_check_compiler_functionality)
    # If the result variable is already defined, this has already been called once, so don't check agian.
    if(DEFINED FLAMEGPU_CheckCompilerFunctionality_RESULT)
        return()
    endif()

    # Ensure that required languages are avialable.
    # Enabling the languages must however be perfomed in file scope, so do this in the root cmake after docs only checks have been found.
    include(CheckLanguage)
    check_language(CXX)
    if(NOT CMAKE_CXX_COMPILER)
        message(WARNING "CXX Language Support Not Found")
        set(FLAMEGPU_CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
        return()
    endif()
    enable_language(CXX)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA Language Support Not Found")
        set(FLAMEGPU_CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
        return()
    endif()
    enable_language(CUDA)

    # We need c++17 std::filesytem, but not all compilers which claim to implement c++17 provide filesystem (GCC 7)
    if(NOT DEFINED CUDA_STD_FILESYSTEM)
        # Disable CMAKE_CUDA_ARCHTIECTURES if not already controlled. This is scoped to the function so safe to control.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "OFF")
        endif()
        try_compile(
            CUDA_STD_FILESYSTEM
            "${CMAKE_CURRENT_BINARY_DIR}/try_compile"
            "${CMAKE_CURRENT_LIST_DIR}/CheckCompilerFunctionality/CheckStdFilesystem.cu"
            CXX_STANDARD 17
            CUDA_STANDARD 17
            CXX_STANDARD_REQUIRED "ON"
        )
    endif()
    # If an error occured while building the <filesystem> snippet, report a warning
    if(NOT CUDA_STD_FILESYSTEM)
        # If the GCC versions is known to be bad, give an appropriate error
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.1)
            message(WARNING
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not provide <std::filesystem> even in --std=c++17 mode.\n"
            " Please use GCC >= 8.1.\n"
            " \n")
        else()
            # If the gcc version is not a known problem, emit a generic error.
            message(WARNING
            "<std::filesystem> error with ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} and ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
        endif()
        # Set the result variable to a false-like value
        set(FLAMEGPU_CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
        return()
    endif()
    
    # Original releases of GCC 10.3.0 and 11.1.0 included a bug preventing the use of <chrono> in <nvcc>.
    # This was patched in subsequent versions, and backported in the release branches, but the broken version is still distributed in some cases (i.e. Ubuntu 20.04, but not 21.04).
    # See https://github.com/FLAMEGPU/FLAMEGPU2/issues/575,  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102 
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Try to compile the test case file for inclusion of chrono.
        if(NOT DEFINED GCC_CUDA_STDCHRONO)
            # Disable CMAKE_CUDA_ARCHTIECTURES if not already controlled. This is scoped to the function so safe to control.
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
                set(CMAKE_CUDA_ARCHITECTURES "OFF")
            endif()
            try_compile(
                GCC_CUDA_STDCHRONO
                "${CMAKE_CURRENT_BINARY_DIR}/try_compile"
                "${CMAKE_CURRENT_LIST_DIR}/CheckCompilerFunctionality/CheckStdChrono.cu"
                CXX_STANDARD 17
                CUDA_STANDARD 17
                CXX_STANDARD_REQUIRED "ON"
            )
        endif()
        # If an error occured while building the <chrono> snippet, report a warning
        if(NOT GCC_CUDA_STDCHRONO)
            # If the GCC versions is known to be bad, give an appropriate error
            if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 10.3.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 11.1.0)
                message(WARNING 
                "GCC ${CMAKE_CXX_COMPILER_VERSION} is incompatible with CUDA due to a bug in the <chrono> implementation.\n"
                " Please use an alternative GCC, or a patched version of GCC ${CMAKE_CXX_COMPILER_VERSION}.\n"
                " \n"
                " See the following for more information:\n"
                " https://github.com/FLAMEGPU/FLAMEGPU2/issues/575\n"
                " https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102\n")
            else()
                # If the gcc version is not a known problem, emit a generic error.
                message(WARNING
                "<std::chrono> not usable with ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} and ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
            endif()
            # Set the result variable to a false-like value
            set(FLAMEGPU_CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
            return()
        endif()
    endif()

    # GCC 9 + CUDA 11.0 + std=c++17 errors when attempting to compile std::vector<std::tuple<>>::push_back.
    # This is the only known bad combination, but let's check more often just in case.
    # We no longer use this (intentionally) so if the error occurs emit a warning but not error?
    # See https://github.com/FLAMEGPU/FLAMEGPU2/issues/575,  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102 
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Try to compile the test case file for inclusion of chrono.
        if(NOT DEFINED GCC_CUDA_VECTOR_TUPLE_PUSHBACK)
            # Disable CMAKE_CUDA_ARCHTIECTURES if not already controlled. This is scoped to the function so safe to control.
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
                set(CMAKE_CUDA_ARCHITECTURES "OFF")
            endif()
            try_compile(
                GCC_CUDA_VECTOR_TUPLE_PUSHBACK
                "${CMAKE_CURRENT_BINARY_DIR}/try_compile"
                "${CMAKE_CURRENT_LIST_DIR}/CheckCompilerFunctionality/CheckVectorTuplePushBack.cu"
                CXX_STANDARD 17
                CUDA_STANDARD 17
                CXX_STANDARD_REQUIRED "ON"
            )
        endif()
        # If an error occured while building MWE emit a dev warning. 
        if(NOT GCC_CUDA_VECTOR_TUPLE_PUSHBACK)
            # If the GCC versions is known to be bad, give an appropriate error
            message(AUTHOR_WARNING
                "std::vector<std::tuple<>>::push_back cannot be compiled with ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} and ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} with --std=c++17.\n"
                "Consider using a different ${CMAKE_CUDA_COMPILER_ID} and ${CMAKE_CXX_COMPILER_ID} combination as errors may be encountered.\n"
                "See https://github.com/FLAMEGPU/FLAMEGPU2/issues/650")
            # Not erroring, so don't change the output value, just emit the above developer warning
        
        # The compilation error is somewhat tempremental, so always emit a warning for the known bad combination
        elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.1" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9" AND CMAKE_CXX_COMILER_VERSION VERSION_LESS "10")
            message(AUTHOR_WARNING 
                "CUDA 11.0 with g++ 9 in c++17 mode may encounter compiler segmentation faults with 'std::vector<std::tuple<...>>::push_back'.\n"
                "Consider using CUDA 11.1+ or gcc 8 to avoid potential issues.\n"
                "See https://github.com/FLAMEGPU/FLAMEGPU2/issues/650")
        endif()
    endif()

    # If we made it this far, set the result variable to be truthy
    set(FLAMEGPU_CheckCompilerFunctionality_RESULT "YES" PARENT_SCOPE)
endfunction()


# Call the function imediately, so the file only needs to be included. 
flamegpu_check_compiler_functionality()
