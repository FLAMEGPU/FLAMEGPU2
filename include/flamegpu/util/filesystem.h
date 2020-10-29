#ifndef INCLUDE_FLAMEGPU_UTIL_FILESYSTEM_H_
#define INCLUDE_FLAMEGPU_UTIL_FILESYSTEM_H_

// If earlier than VS 2019
#if defined(_MSC_VER) && _MSC_VER < 1920
#include <filesystem>
using std::tr2::sys::exists;
using std::tr2::sys::path;
using std::tr2::sys::create_directory;
#else
// VS2019 requires this macro, as building pre c++17 cant use std::filesystem
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
using std::experimental::filesystem::v1::exists;
using std::experimental::filesystem::v1::path;
using std::experimental::filesystem::v1::create_directory;
#endif

namespace util {
namespace filesystem {
/**
 * Creates the directory pointed to by path
 * Regular directory creation fails to create directories when multiple layers are missing, this is why we recurse
 * @throw May throw implementation specific exceptions, e.g. if the path is invalid
 */
inline void recursive_create_dir(const path &dir) {
    if (::exists(dir)) {
        return;
    }
    path parent_dir = absolute(dir).parent_path();
    if (!::exists(parent_dir)) {
        recursive_create_dir(parent_dir);
    }
    create_directory(dir);
}

}  // namespace filesystem
}  // namespace util

#endif  // INCLUDE_FLAMEGPU_UTIL_FILESYSTEM_H_
