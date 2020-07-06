###################################
# Flamegpu2 Visualisation Library #
###################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)
cmake_policy(SET CMP0079 NEW)

# If overridden by the user, attempt to use that
if (VISUALISATION_ROOT)
    message("VisualisationRoot override")
    add_subdirectory(${VISUALISATION_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/_deps/flamegpu2_visualiser-build EXCLUDE_FROM_ALL)
    # Set locally and for parent scope, which are mutually exclusive
    set(VISUALISATION_BUILD ${CMAKE_CURRENT_BINARY_DIR}/_deps/flamegpu2_visualiser-build CACHE INTERNAL "flamegpu2_visualiser_BINARY_DIR")
else()
    # Otherwise download.
    FetchContent_Declare(
        flamegpu2_visualiser
        GIT_REPOSITORY https://github.com/FLAMEGPU/FLAMEGPU2_visualiser.git
        GIT_TAG        master
        GIT_PROGRESS   ON
        # UPDATE_DISCONNECTED   ON
        )
        FetchContent_GetProperties(flamegpu2_visualiser)
    if(NOT flamegpu2_visualiser_POPULATED)
        FetchContent_Populate(flamegpu2_visualiser)
    
        add_subdirectory(${flamegpu2_visualiser_SOURCE_DIR} ${flamegpu2_visualiser_BINARY_DIR} EXCLUDE_FROM_ALL)
        
        # Set locally and for parent scope, which are mutually exclusive
        set(VISUALISATION_ROOT ${flamegpu2_visualiser_SOURCE_DIR} CACHE INTERNAL "flamegpu2_visualiser_SOURCE_DIR")
        set(VISUALISATION_BUILD ${flamegpu2_visualiser_BINARY_DIR} CACHE INTERNAL "flamegpu2_visualiser_BINARY_DIR")
    endif()
endif()
