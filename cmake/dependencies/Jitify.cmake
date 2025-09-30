##########
# Jitify #
##########

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules/ ${CMAKE_MODULE_PATH})
include(FetchContent)

# Change the source-dir to allow inclusion via jitify/jitify.hpp rather than jitify.hpp
# Specify an invalid SOURCE_SUBDIR to prevent FetchContent_MakeAvailable from adding the CMakeLists.txt
FetchContent_Declare(
    jitify
    GIT_REPOSITORY https://github.com/NVIDIA/jitify.git
    GIT_TAG        44e978b21fc8bdb6b2d7d8d179523c8350db72e5  # jitify2 branch on 2025-08-23, pre windows patches
    SOURCE_DIR     ${FETCHCONTENT_BASE_DIR}/jitify-src/jitify
    GIT_PROGRESS   ON
    # UPDATE_DISCONNECTED   ON
    SOURCE_SUBDIR "do_not_use_add_subirectory"
)
FetchContent_GetProperties(jitify)
# Download content but do not use add_subdirectory
FetchContent_MakeAvailable(jitify)

# Apply patches to jitify2 for msvc support prior to https://github.com/NVIDIA/jitify/pull/146 being merged
# patch generated from jitify fork/branch, via `git format-patch jitify2 --stdout > jitify2-pr-146.patch`
# Does not use fetch content PATCH_COMMAND as it was unreliable / problematic on windows for prior rapidjson patching
# This will silently not apply the patch if it is not applicable, which supports repeated fetching but may lead to silent failures
# Check if the patch is applicable
execute_process(
    COMMAND git apply --check ${CMAKE_CURRENT_LIST_DIR}/patches/jitify2-pr-146.patch
    WORKING_DIRECTORY "${jitify_SOURCE_DIR}"
    RESULT_VARIABLE jitify_patch_check_result
    OUTPUT_QUIET
    ERROR_QUIET
)
# If applicable, apply the patch
if (jitify_patch_check_result EQUAL 0)
    message(CHECK_START  "Patching jitify #146")
    execute_process(
        COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/patches/jitify2-pr-146.patch
        WORKING_DIRECTORY "${jitify_SOURCE_DIR}"
        RESULT_VARIABLE jifity_patch_apply_result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    # If the patch failed emit a warning but allow cmake to progress. This should not occur given the previous check.
    if (jifity_patch_apply_result EQUAL 0)
        message(CHECK_PASS "done")
    else()
        message(CHECK_FAIL "patching failed")
    endif()
    unset(jifity_patch_apply_result)
endif()
unset(jitify_patch_check_result)

# Jitify is not a cmake project, so cannot use add_subdirectory, use custom find_package.
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${jitify_SOURCE_DIR}/..")

# Always find the package, even if jitify is already populated.
find_package(Jitify REQUIRED)

# Mark some CACHE vars advanced for a cleaner GUI
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_JITIFY)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED) 
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_JITIFY) 