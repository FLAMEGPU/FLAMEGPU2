# Define a cmake function which checks that CUDA and the host compiler are functional.
function(CheckCompilerFunctionality)
    # If the result variable is already defined, this has already been called once, so don't check agian.
    if(DEFINED CheckCompilerFunctionality_RESULT)
        return()
    endif()

    # Ensure that required languages are avialable.
    # Enabling the languages must however be perfomed in file scope, so do this in the root cmake after docs only checks have been found.
    include(CheckLanguage)
    check_language(CXX)
    if(NOT CMAKE_CXX_COMPILER)
        message(WARNING "CXX Language Support Not Found")
        set(CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
        return()
    endif()
    enable_language(CXX)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA Language Support Not Found")
        set(CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
        return()
    endif()
    enable_language(CUDA)
    
    # Original releases of GCC 10.3.0 and 11.1.0 included a bug preventing the use of <chrono> in <nvcc>.
    # This was patched in subsequent versions, and backported in the release branches, but the broken version is still distributed in some cases (i.e. Ubuntu 20.04, but not 21.04).
    # See https://github.com/FLAMEGPU/FLAMEGPU2/issues/575,  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102 
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Try to compile the test case file for inclusion of chrono.
        if(NOT DEFINED GCC_CUDA_STDCHRONO)
            # CUDA must be available.
            enable_language(CUDA)
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
            set(CheckCompilerFunctionality_RESULT "NO" PARENT_SCOPE)
            return()
        endif()
    endif()

    # If we made it this far, set the result variable to be truthy
    set(CheckCompilerFunctionality_RESULT "YES" PARENT_SCOPE)
endfunction()


# Call the function imediately, so the file only needs to be included. 
CheckCompilerFunctionality()
