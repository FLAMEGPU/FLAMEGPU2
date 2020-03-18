# Pull in and build visualiser
# Creates target FLAMEGPU2_Visualiser
MACRO (download_visualiser)
    if (VISUALISATION_ROOT)
        message("VisualisationRoot override")
        add_subdirectory(${VISUALISATION_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/visualiser/build EXCLUDE_FROM_ALL)
    else()
        configure_file(${FLAMEGPU_ROOT}/cmake/visualiser/CMakeLists.txt.in ${CMAKE_CURRENT_BINARY_DIR}/visualiser/download/CMakeLists.txt)
        # Run CMake generate
        execute_process(
            COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/visualiser/download
            )
        if (result)
            message(WARNING 
                    "CMake step for visualiser failed: ${result}\n")
        endif()
        # Run CMake build (this only downloads, it is built at build time)
        execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/visualiser/download
        )
        if (result)
            message(WARNING 
                    "Download step for visualiser-build failed: ${result}\n"
                    "Attempting to continue\n")
        endif()
        add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/visualiser/src ${CMAKE_CURRENT_BINARY_DIR}/visualiser/build EXCLUDE_FROM_ALL)
        # Called from build-fgpu2, need it in scope of common.cmake:addFGPUExec
        set(VISUALISATION_ROOT ${CMAKE_CURRENT_BINARY_DIR}/visualiser/src)
        set(VISUALISATION_ROOT ${CMAKE_CURRENT_BINARY_DIR}/visualiser/src PARENT_SCOPE)
    endif()
    set(VISUALISATION_BUILD ${CMAKE_CURRENT_BINARY_DIR}/visualiser/build)
    set(VISUALISATION_BUILD ${CMAKE_CURRENT_BINARY_DIR}/visualiser/build PARENT_SCOPE)
ENDMACRO()