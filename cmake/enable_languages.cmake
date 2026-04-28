# CMake >= 3.25.2 for CUDA C++20 detection. >= 3.21 for HIP support.
cmake_minimum_required(VERSION 3.25.2...4.3.0 FATAL_ERROR)

# Module for abstracting selection of CUDA or HIP or DOCS only builds
# Todo:
# - Better cache variable than FLAMEGPU_GPU?
# - if no FLAMEGPU_GPU value set and CUDA not found, automatically try hip?
# - Figure out how to nicely handle this for projects which have CUDA enabled (i.e. template examples...) if HIP was requested or used
# - Figure out how standalone examples which add_project the main FLAMEGPU source should work, so that they use the same value as the FLAMEGPU target.
# - Figure out how this might work in a future find_package(FLAMEGPU) to avoid future CMake breaks?
# - Test this on all platforms with loads of diff compiler combinations :(

# Only define the macro once
include_guard(GLOBAL)
include(CheckLanguage)
include(CheckSourceCompiles)

# Store the location of this file, so relative paths can be used inside macros (which exist in the scope of the caller)
set(_FLAMEGPU_ENABLE_LANGUAGES_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Define the minimum supported CUDA and HIP versions
set(MINIMUM_SUPPORTED_CUDA_VERSION 12.4)
set(MINIMUM_SUPPORTED_HIP_VERSION 7.0)

# Set the std which is used in compilation testing examples
set(_flamegpu_cxx_std 20)

# Define a cache variable FLAMEGPU_GPU controlling the GPU API to use, defaulting to CUDA in the GUI.
set(_FLAMEGPU_GPU_OPTIONS "CUDA;HIP;OFF")
set(FLAMEGPU_GPU "CUDA" CACHE STRING "The GPU API to use. Choose from ${_FLAMEGPU_GPU_OPTIONS}. Use OFF for a documentation-only build." )
set_property(CACHE FLAMEGPU_GPU PROPERTY STRINGS ${_FLAMEGPU_GPU_OPTIONS})
# Validate that FLAMEGPU_GPU is set to an allowed value
if(NOT "${FLAMEGPU_GPU}" IN_LIST _FLAMEGPU_GPU_OPTIONS)
    message(FATAL_ERROR "\"FLAMEGPU_GPU\" has an invalid value: \"${FLAMEGPU_GPU}\". Select from: ${_FLAMEGPU_GPU_OPTIONS}")
endif()
unset(_FLAMEGPU_GPU_OPTIONS)

# Define a macro (enable_language should not be called from a function) which enables languages based on current cache variables, and will error if a language is already defined that conflicts with FLAMEGPU_GPU
macro(flamegpu_enable_languages)
    # Raise an error if project() has not yet been called, as enable_language cannot be called without without a project()
    if (NOT PROJECT_NAME)
        message(FATAL_ERROR "flamegpu_enable_languages must be called after a call to 'project()'")
    endif()

    # Check for Host and Device compiler support if not in an intentional documentation-only build
    if (NOT ${FLAMEGPU_GPU} STREQUAL "OFF")
        # Get the currently enabled languages, so that if a parent project has HIP enabled but FLAMEGPU is configured for CUDA we can raise an appropriate warning/error
        get_property(_enabled_langs GLOBAL PROPERTY ENABLED_LANGUAGES)

        # Raise an error if CUDA requested but HIP is an enabled language
        if(${FLAMEGPU_GPU} STREQUAL "CUDA" AND "HIP" IN_LIST _enabled_langs)
            message(FATAL_ERROR "HIP is enabled in this project, but FLAMEGPU_GPU=${FLAMEGPU_GPU}")
        endif()

        # Raise an error if CUDA requested but HIP is an enabled language
        if(${FLAMEGPU_GPU} STREQUAL "HIP" AND "CUDA" IN_LIST _enabled_langs)
            message(FATAL_ERROR "CUDA is enabled in this project, but FLAMEGPU_GPU=${FLAMEGPU_GPU}")
        endif()

        # Enable C and CXX, which will error if neither can be enabled.
        enable_language(C)
        enable_language(CXX)

        # Ensure that the found host compiler is c++20 compatible, raising a FATAL ERROR if not.
        _flamegpu_check_source_compiles_cxx_std()

        # Check for and enable CUDA if it was requested
        if (${FLAMEGPU_GPU} STREQUAL "CUDA")
            # Check for CUDA support
            check_language(CUDA)
            if (CMAKE_CUDA_COMPILER)
                # enable CUDA if it was found
                enable_language(CUDA)
                # Set the user-provided or flamegpu-default CMAKE_CUDA_ARCHITECTURES now the CUDA version is known
                # flamegpu_set_gpu_architectures()
            else()
                # If CUDA could not be found, a fatal error is raised
                # The error message varies in some cases, as we do not know exactly why CUDA support was not found, but can try to be helpful for some (msvc related) issues:
                if (MSVC)
                    # MSVC >= 1941 requires CUDA >= 12.4, but we target/support >= 12.0, so issue a specific message
                    if (MSVC_VERSION VERSION_GREATER_EQUAL "1941")
                        # Modify the message based on CMake < 3.29.4
                        set(MSG_CUDAFLAGS "- Downgrade to MSVC 1940 and set the CUDAFLAGS environment variable to contain '-allow-unsupported-compiler'")
                        if(CMAKE_VERSION VERSION_LESS "3.29.4")
                            set(MSG_CUDAFLAGS "- Downgrade to MSVC 1940, upgrade to CMake >= 3.29.4 and set the CUDAFLAGS environment variable to contain '-allow-unsupported-compiler'")
                        endif()
                        message(FATAL_ERROR
                            " CUDA Language Support Not Found (with MSVC ${MSVC_VERSION} >= 1941 and CMake ${CMAKE_VERSION}))\n"
                            " \n"
                            " The MSVC STL included with MSVC >= 1941 requires CUDA >= 12.4\n"
                            " If you have CUDA 12.0-12.4 installed you must either:\n"
                            " \n"
                            "  - Upgrade to CUDA >= 12.4\n"
                            "  - Downgrade to MSVC <= 1939\n"
                            "  ${MSG_CUDAFLAGS}\n"
                            "  - Set the NVCC_PREPEND_FLAGS environment variable to include '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -allow-unsupported-compiler' for configuration and compilation\n"
                            " \n"
                            " You must clear the CMake cache before reconfiguring this project\n"
                        )
                    # MSVC == 1940 and CUDA <= 12.3 requires -allow-unsupported-compiler (and therefore CMake >= 3.29.4)
                    elseif(MSVC_VERSION VERSION_EQUAL "1940")
                        # Modify the message based on CMake < 3.29.4
                        set(MSG_CUDAFLAGS "- Set the CUDAFLAGS environment variable to contain '-allow-unsupported-compiler'")
                        if(CMAKE_VERSION VERSION_LESS "3.29.4")
                            set(MSG_CUDAFLAGS "- Upgrade to CMake >= 3.29.4 and set the CUDAFLAGS environment variable to contain '-allow-unsupported-compiler'")
                        endif()
                        message(FATAL_ERROR
                            " CUDA Language Support Not Found (with MSVC ${MSVC_VERSION} == 1940 and CMake ${CMAKE_VERSION})\n"
                            " \n"
                            " CUDA <= 12.3 did not expect MSVC >= 1940 to part of Visual Studio 2022\n"
                            " If you have CUDA 12.0-12.3 installed you must either:\n"
                            " \n"
                            "  - Upgrade to CUDA >= 12.4\n"
                            "  - Downgrade to MSVC <= 1939\n"
                            "  ${MSG_CUDAFLAGS}\n"
                            "  - Set the NVCC_PREPEND_FLAGS environment variable to include '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -allow-unsupported-compiler' for configuration and compilation\n"
                            " \n"
                            " You must clear the CMake cache before reconfiguring this project\n"
                            " \n"
                        )
                    endif()
                endif()

                # If any msvc specific fatal errors were not raised, raise the generic error.
                message(FATAL_ERROR 
                "  CUDA language support could not be found (with FLAMEGPU_GPU=${FLAMEGPU_GPU}).\n"
                "  Please ensure CUDA >= ${MINIMUM_SUPPORTED_CUDA_VERSION} is installed and discoverable by CMake, or request and alternative GPU backend via FLAMEGPU_GPU")
            endif()

            # Ensure that the found CUDA version is at least the minimum required by FLAMEGPU, otherwise raise an error.
            # Note: this is a breaking change to CMake behaviour (for docs-only builds), must set FLAMEGPU_GPU=OFF instead.
            if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
                message(FATAL_ERROR "CUDA ${MINIMUM_SUPPORTED_CUDA_VERSION} or greater is required for compilation with FLAMEGPU_GPU=${FLAMEGPU_GPU}")
            endif()

        # Check for and enable HIP if it was requested
        elseif(${FLAMEGPU_GPU} STREQUAL "HIP")
            # Check for HIP support
            check_language(HIP)
            if (CMAKE_HIP_COMPILER)
                # Enable HIP if it was found
                enable_language(HIP)
                # Set the user-provided or flamegpu-default CMAKE_HIP_ARCHITECTURES now the CUDA version is known
                # flamegpu_set_gpu_architectures()
            else()
                # If HIP could not be found, a fatal error is raised
                message(FATAL_ERROR 
                "  HIP language support could not be found (with FLAMEGPU_GPU=${FLAMEGPU_GPU}).\n"
                "  Please ensure HIP/ROCm >= ${MINIMUM_SUPPORTED_HIP_VERSION} is installed and discoverable by CMake, or request and alternative GPU backend via FLAMEGPU_GPU")
            endif()

            # Ensure that the found HIP version is at least the minimum required by FLAMEGPU, otherwise raise an error.
            if (CMAKE_HIP_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_HIP_VERSION})
                message(FATAL_ERROR "HIP ${MINIMUM_SUPPORTED_HIP_VERSION} or greater is required for compilation with FLAMEGPU_GPU=${FLAMEGPU_GPU}")
            endif()
        endif()

        # Emit warnings for known compiler versions that may cause errors, but that are not detected by compilation testing
        _flamegpu_languages_warn_win_cu124()

        # Double check cxx20 is Ok with hip/cuda
        _flamegpu_check_source_compiles_cuda_std()
        _flamegpu_check_source_compiles_hip_std()

        # Perform various compiler support checks, emitting WARNINGs or FATAL_ERRORs depending on the severity
        _flamegpu_check_source_compiles_cxx_filesystem()
        _flamegpu_check_source_compiles_cuda_chrono()
        _flamegpu_check_source_compiles_cuda_vector_tuple()
        _flamegpu_check_source_compiles_cuda_source_location()
        _flamegpu_check_source_compilers_hip_cxx_xhip()

        # Set cache variables which can only be determined by compilation 
        _flamegpu_check_source_compiles_file_offset_bits_64()

    else() # (${FLAMEGPU_GPU} STREQUAL "OFF")
        # TODO: Set this in parent scope or cache? if necessary, can probably just check for `CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER` instead.
        set(DOCS_ONLY_BUILD ON)
    endif()

    # Success, only print once per cmake configuration
    get_property(_already_printed GLOBAL PROPERTY FLAMEGPU_ENABLE_LANGUAGES_PRINT_ONCE)
    if(NOT _already_printed)
        message(STATUS "FLAMEGPU_GPU=${FLAMEGPU_GPU}")
        if (CMAKE_CUDA_COMPILER_LOADED)
            message(STATUS "CUDA Enabled")
            message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        elseif(CMAKE_HIP_COMPILER_LOADED)
            message(STATUS "HIP Enabled")
            message(STATUS "HIP Architectures: ${CMAKE_HIP_ARCHITECTURES}")
        else() # elseif(NOT (CMAKE_CUDA_COMPILER_LOADED OR CMAKE_HIP_COMPILER_LOADED))
            message(STATUS "Documentation only - NOT (CMAKE_CUDA_COMPILER_LOADED OR CMAKE_HIP_COMPILER_LOADED) / FLAMEGPU_GPU=OFF")
        endif()
    endif()
    set_property(GLOBAL PROPERTY FLAMEGPU_ENABLE_LANGUAGES_PRINT_ONCE TRUE)
endmacro()


# Internal function to check if host compiler builds with c++20. Raises a FATAL ERROR if compilation fails.
function(_flamegpu_check_source_compiles_cxx_std)
    # Return early if CXX is not loaded
    if (NOT CMAKE_CXX_COMPILER_LOADED)
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Compile the following snippet - anything should be fine to trigger a strict cxx standard error?
    check_source_compiles(CXX [[
        int main() { return 0; } 
    ]] _FLAMEGPU_CHECK_CXX_STD_V0)

    if (NOT _FLAMEGPU_CHECK_CXX_STD_V0)
        message(FATAL_ERROR
            "CXX compiler unable to compile with CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD}"
        )
    endif()
endfunction()


# Internal function to check if cuda compiler builds with c++20. Raises a FATAL ERROR if compilation fails.
function(_flamegpu_check_source_compiles_cuda_std)
    # Return early if CUDA is not loaded
    if (NOT CMAKE_CUDA_COMPILER_LOADED)
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Compile the following snippet - anything should be fine to trigger a strict cxx standard error?
    check_source_compiles(CUDA [[
        int main() { return 0; } 
    ]] _FLAMEGPU_CHECK_CUDA_STD_V0)

    if (NOT _FLAMEGPU_CHECK_CUDA_STD_V0)
        message(FATAL_ERROR
            "CUDA compiler unable to compile with CMAKE_CUDA_STANDARD ${CMAKE_CUDA_STANDARD}"
        )
    endif()
endfunction()

# Internal function to check if hip compiler builds with c++20. Raises a FATAL ERROR if compilation fails.
function(_flamegpu_check_source_compiles_hip_std)
    # Return early if HIP is not loaded
    if (NOT CMAKE_HIP_COMPILER_LOADED)
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    # Compile the following snippet - anything should be fine to trigger a strict cxx standard error?
    check_source_compiles(HIP [[
        int main() { return 0; } 
    ]] _FLAMEGPU_CHECK_HIP_STD_V0)

    if (NOT _FLAMEGPU_CHECK_HIP_STD_V0)
        message(FATAL_ERROR
            "HIP compiler unable to compile with CMAKE_HIP_STANDARD ${CMAKE_HIP_STANDARD}"
        )
    endif()
endfunction()

# Internal function to check if c++17+ std::filesystem compilation succeed, raising a FATAL_ERROR if not
# This was a known issue for GCC 7 which allows c++17 to be specified, but does not implement std::filesystem
# This could probably be removed with the move to c++20
# Note: if the source snippet is changed, the internal variable name should be increased so that the snippet gets re-compiled.
function(_flamegpu_check_source_compiles_cxx_filesystem)
    # Return early if neither CUDA nor HIP are loaded
    if (NOT (CMAKE_CUDA_COMPILER_LOADED OR CMAKE_HIP_COMPILER_LOADED))
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Compile the following snippet
    check_source_compiles(CXX [[
        #include <filesystem>
        int main() { return 0; } 
    ]] _FLAMEGPU_CHECK_CXX_FILESYSTEM_V0)

    if (NOT _FLAMEGPU_CHECK_CXX_FILESYSTEM_V0)
        # If the GCC versions is known to be bad, give an appropriate error
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.1)
            message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not provide <std::filesystem> even in --std=c++17 mode.\n"
            " Please use GCC >= 8.1.\n"
            " \n")
        else()
            # If the gcc version is not a known problem, emit a generic error.
            message(FATAL_ERROR
                "<std::filesystem> error with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}."
            )
        endif()
    endif()
endfunction()

# Internal function to check for a known issue preventing use of <chrono> within nvcc with some builds of GCC 10.3.0/11.1.0.
# See https://github.com/FLAMEGPU/FLAMEGPU2/issues/575,  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102 
# Note: if the source snippet is changed, the internal variable name should be increased so that the snippet gets re-compiled.
function(_flamegpu_check_source_compiles_cuda_chrono)
    # Return early if not using CUDA with GCC
    if (NOT (CMAKE_CUDA_COMPILER_LOADED AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CUDA_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Compile the following snippet
    check_source_compiles(CUDA [[
        #include <chrono>
        int main() { return 0; }
    ]] _FLAMEGPU_CHECK_GCC_CUDA_CHRONO_V0)

    if (NOT _FLAMEGPU_CHECK_GCC_CUDA_CHRONO_V0)
        # If the GCC versions is known to be bad, give an appropriate error
        if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 10.3.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 11.1.0)
            message(FATAL_ERROR 
                "GCC ${CMAKE_CXX_COMPILER_VERSION} is incompatible with CUDA due to a bug in the <chrono> implementation.\n"
                " Please use an alternative GCC, or a patched version of GCC ${CMAKE_CXX_COMPILER_VERSION}.\n"
                " \n"
                " See the following for more information:\n"
                " https://github.com/FLAMEGPU/FLAMEGPU2/issues/575\n"
                " https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100102\n"
            )
        else()
            # If the gcc version is not a known problem, emit a generic error.
            message(FATAL_ERROR
                "<std::chrono> not usable with ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} and ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}."
            )
        endif()
    endif()
endfunction()


# Internal function to check for a known issue compile std::vector<std::tuple<>>::push_back, which previously effected GCC 9 + CUDA 11.0 + std=c++17 .
# This is the only known bad combination, and is no longer supported for FLAMEGPU but check anyway
# See hSee https://github.com/FLAMEGPU/FLAMEGPU2/issues/650
# Note: if the source snippet is changed, the internal variable name should be increased so that the snippet gets re-compiled.
function(_flamegpu_check_source_compiles_cuda_vector_tuple)
    # Return early if not using CUDA with GCC
    if (NOT (CMAKE_CUDA_COMPILER_LOADED AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CUDA_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Compile the following snippet
    check_source_compiles(CUDA [[
        #include <tuple>
        #include <vector>
        int main (int argc, char * argv[]) {
            std::vector<std::tuple<float>> v;
            std::tuple<float> t = {1.f};
            v.push_back(t);  // segmentation fault
        }
    ]] _FLAMEGPU_CHECK_GCC_CUDA_VECTOR_TUPLE_V0)

    if (NOT _FLAMEGPU_CHECK_GCC_CUDA_VECTOR_TUPLE_V0)
        message(WARNING
            "std::vector<std::tuple<>>::push_back cannot be compiled with ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} and ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} with --std=c++${_flamegpu_cxx_std}.\n"
            "Consider using a different ${CMAKE_CUDA_COMPILER_ID} and ${CMAKE_CXX_COMPILER_ID} combination as errors may be encountered.\n"
            "See https://github.com/FLAMEGPU/FLAMEGPU2/issues/650"
        )
    endif()
endfunction()

# Internal function to ensure that <source_location> can be used with the host compiler, i.e. GCC >= 11
# Note: if the source snippet is changed, the internal variable name should be increased so that the snippet gets re-compiled.
function(_flamegpu_check_source_compiles_cuda_source_location)
    # Return early if not using CUDA with GCC
    if (NOT (CMAKE_CUDA_COMPILER_LOADED AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CUDA_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Compile the following snippet
    check_source_compiles(CUDA [[
        #include <cstdio>
        #include <source_location>
        void print_file_line(const std::source_location loc = std::source_location::current()) {
            printf("%s:%d\n", loc.file_name(), loc.line());
        }
        int main (int argc, char * argv[]) {
            print_file_line();
        }
    ]] _FLAMEGPU_CHECK_GCC_CUDA_SOURCE_LOCATION_V0)

    if (NOT _FLAMEGPU_CHECK_GCC_CUDA_SOURCE_LOCATION_V0)
        message(FATAL_ERROR
            "std::source_location::current() cannot be compiled with ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} and ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} with --std=c++${_flamegpu_cxx_std}.\n"
            "GCC >= 11 is required, with CUDA >= 12.2"
        )
    endif()
endfunction()

# Internal function that emits a warning if compiling a CXX file with -x hip fails
# hip::device (for rocm 7.2.0 atleast) includes '-x hip' in INTERFACE_COMPILE_OPTIONS from lib/cmake/hip-config-amd.cmake
# This is applied to all targets regardless of the langauge, so if the CXX (or C) compiler is not hip aware, errors will likely occur.
# This is just a warning, not an error incase HIP behaviour becomes sane in the future, and this check occurs before the call to find_package(HIP)
function(_flamegpu_check_source_compilers_hip_cxx_xhip)
    # Do nothing if on MSVC, or if not using HIP
    if(MSVC OR NOT CMAKE_HIP_COMPILER_LOADED)
        return()
    endif()

    # Early exit if this has already been ran
    if(DEFINED CACHE{_FLAMEGPU_CHECK_HIP_CXX_X_HIP_V0})
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    # set the -x hip flag which will probably cause errors
    set(CMAKE_REQUIRED_FLAGS "-x hip")
    # Only build a static library, as -x hip will be an error if passed to the linker.
    # this is a check_source_compilers limitation, try_compile could be used instead.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

    # Compile the following snippet
    check_source_compiles(CXX [[
        int main (int argc, char * argv[]) {
            return 0;
        }
    ]] _FLAMEGPU_CHECK_HIP_CXX_X_HIP_V0)

    if (NOT _FLAMEGPU_CHECK_HIP_CXX_X_HIP_V0)
        message(WARNING
            "  ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} fails to compile with '-x hip'\n"
            "  Compilation errors are likely to occur due to INTERFACE_COMPILE_OPTIONS from hip::device likely including '-x hip'\n"
            "  Consider using an alternate CXX compiler which is hip-aware, via CMAKE_CXX_COMPILER such as hipcc or amdclang++\n"
        )
    endif()

endfunction()

# Internal function to check if the _FILE_OFFSET_BITS=64 is required for Jitify2 or not on linux. 
# Jitify2 on linux requires uses F_OFD_SETLKW or -D_FILE_OFFSET_BITS=64. 
# Cannot detect the presence of this from CMAKE_SYSTEM_VERSION when in a container, so try to compile with it and if an error occurs then we must define this option.
# If required, sets a cmake cache internal variable FLAMEGPU__FILE_OFFSET_BITS_64_REQUIRED to true
# Note: if the source snippet is changed, the internal variable name should be increased so that the snippet gets re-compiled.
function(_flamegpu_check_source_compiles_file_offset_bits_64)
    # Do nothing if on MSVC, or if not using CUDA
    if(MSVC OR NOT CMAKE_CUDA_COMPILER_LOADED)
        return()
    endif()

    # Early exit if this has already been ran
    if(DEFINED CACHE{FLAMEGPU__FILE_OFFSET_BITS_64_REQUIRED})
        return()
    endif()

    # Set local scope variables that impact the check 
    set(CMAKE_CXX_STANDARD ${_flamegpu_cxx_std})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Compile the following snippet
    check_source_compiles(CXX [[
        #ifdef __linux__
        #include <fcntl.h>
        #endif  // __linux__

        int main (int argc, char * argv[]) {
            #ifndef F_OFD_SETLKW
            #error F_OFD_SETLKW is not defined; try building with -D_FILE_OFFSET_BITS=64
            #endif  // F_OFD_SETLKW
            return 0;
        }   
    ]] _FLAMEGPU_LINUX_CUDA_F_OFD_SETLKW_V0)

    # Invert the logic to match your required variable name
    if(NOT _FLAMEGPU_LINUX_CUDA_F_OFD_SETLKW_V0)
        set(FLAMEGPU__FILE_OFFSET_BITS_64_REQUIRED TRUE CACHE INTERNAL "if _FILE_OFFSET_BITS=64 required for jitify2")
    else()
        set(FLAMEGPU__FILE_OFFSET_BITS_64_REQUIRED FALSE CACHE INTERNAL "if _FILE_OFFSET_BITS=64 required for jitify2")
    endif()
endfunction()


# Internal function which will emit a single warning per CMake confugration that CUDA < 12.4 on Windows in theory should work, 
# but in practice might not due to compilation errors under c++20 on Windows, which we cannot reliably test fixes for.
function(_flamegpu_languages_warn_win_cu124)
    if (MSVC AND CMAKE_CUDA_COMPILER_LOADED AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4)
        if (NOT MSVC_CUDA_LT_124_SHOWN)
            message(WARNING
                " Parts of flamegpu may fail to build with MSVC and CUDA < 12.4 due to compilation errors under c++20.\n"
                " \n"
                " Please consider upgrading to CUDA >= 12.4."
                " \n"
            )
            set(MSVC_CUDA_LT_124_SHOWN TRUE PARENT_SCOPE )
        endif()
    endif()
endfunction()