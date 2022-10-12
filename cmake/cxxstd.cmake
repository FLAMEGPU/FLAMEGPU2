# Select the CXX standard to use, FLAME GPU 2 is c++17 only
if(NOT FLAMEGPU_CXX_STD)
    # No need to check CMake version, as our minimum (3.18) supports CUDA c++17

    # Check the CUDA version, CUDA 11.0 adds CXX 17 support
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0.0)
        # Fatal Error.
        message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not support -std=c++17")
    endif()

    # Check MSVC version, VS 2017 version 15.3 added /std:c++17 - 1911
    # Inside source code, __STDC_VERSION__ can be used on msvc, which will have a value such as 201710L for c++17
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
            message(FATAL_ERROR "MSVC ${CMAKE_CXX_COMPILER_VERSION} does not support -std=c++17 (>= 19.11 required)")
        endif()
    endif()

    # GCC 8 required for <filesystem>
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.1)
        message(FATAL_ERROR "GCC >= 8.1 required for -std=c++17 <filesystem>")
    endif()

    # Set a cmake variable so this is only calcualted once, and can be applied afterwards.
    set(FLAMEGPU_CXX_STD 17)
endif()

# @future - set this on a per target basis using set_target_properties?
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${FLAMEGPU_CXX_STD})
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${FLAMEGPU_CXX_STD})
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()
