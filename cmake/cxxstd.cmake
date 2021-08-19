# Select the CXX standard to use. 
if(NOT FLAMEGPU_CXX_STD)
    # FLAME GPU is c++14, however due to MSVC 16.10 regressions we build as 17 if possible, else 14. 
    # 14 Support is still required (CUDA 10.x, swig?).
    # Start by assuming both should be availble.
    set(CXX17_SUPPORTED ON)
    # CMake 3.18 adds CUDA CXX 17, 20
    # CMake 3.10 adds CUDA CXX 14
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        # 17 OK
    elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.10)
        # No need for deprecation warning here, already warning about CMAKE < 3.18 being deprecated elsewhere.
        set(CXX17_SUPPORTED OFF)
    else()
        message(FATAL_ERROR "CMAKE ${CMAKE_VERSION} does not support -std=c++14")
    endif()
    # CUDA 11.0 adds CXX 17
    # CUDA 9.0 adds CXX 14
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0.0)
        # 17 is ok.
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0.0)
        # 14 is ok, 17 is not.
        # no need for an extra deprecation warning here, already warning in common about CUDA version.
        set(CXX17_SUPPORTED OFF)
    else()
        # Fatal Error.
        message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not support -std=c++14")
    endif()

    # VS 2019 16.10.0 breaks CXX 14 + cuda. - 1930? 19.29.30037?
    # VS 2017 version 15.3 added /std:c++17 - 1911
    # MSVC VS2015 Update 3 added /std:c++14 >= 1900 && < 1910? 
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.29)
            # 17 required.
            if(NOT CXX17_SUPPORTED)
                message(FATAL_ERROR "MSVC >= 19.29 requires CMake >= 3.18 and CUDA >= 11.0")
            endif()
        elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.11)
            # 17 available?
        elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.10)
            # Emit a deprecation warning for VS20XX that is not c++17 supporting. I think this is VS2017 which will be broken anyway with CUDA 
            message(DEPRECATION "Use of MSVC ${CMAKE_CXX_COMPILER_VERSION} is deprecated. A C++17 compiler (>= 19.11) will be required in a future release.")
            set(CXX17_SUPPORTED OFF)
        else()
            message(FATAL_ERROR "MSVC ${CMAKE_CXX_COMPILER_VERSION} does not support -std=c++14")
        endif()
    endif()

    # gcc supported C++17 since 5, so any version supported by cuda 10+ (no need to check, a configure time error will already occur.)
    # Inside source code, __STDC_VERSION__ can be used on msvc, which will have a value such as 201710L for c++17

    # Set a cmake variable so this is only calcualted once, and can be applied afterwards.
    if(CXX17_SUPPORTED)
        set(FLAMEGPU_CXX_STD 17)
    else()
        set(FLAMEGPU_CXX_STD 14)
    endif()
    # if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    #     # Forward to the parent scope so it persists between calls.
    #     set(FLAMEGPU_CXX_STD ${FLAMEGPU_CXX_STD} PARENT_SCOPE)
    # endif()

    # Emit a developer warning if using CUDA 11.0 and GCC 9 in c++17 mode re std::vector<std::tuple<...>>::push_back
    if( CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0"
        AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.1"
        AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9"
        AND CMAKE_CXX_COMILER_VERSION VERSION_LESS "10"
        AND (FLAMEGPU_CXX_STD EQUAL 17 OR CMAKE_CUDA_STANDARD EQUAL 17))
        # https://github.com/FLAMEGPU/FLAMEGPU2/issues/650
        message(AUTHOR_WARNING 
            "CUDA 11.0 with g++ 9 in c++17 mode may encounter compiler segmentation faults with 'std::vector<std::tuple<...>>::push_back'.\n"
            "Consider using CUDA 11.1+ or gcc 8 to avoid potential issues.\n"
            "See https://github.com/FLAMEGPU/FLAMEGPU2/issues/650 for more information.")
    endif()
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