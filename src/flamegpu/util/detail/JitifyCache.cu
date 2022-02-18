#include "flamegpu/util/detail/JitifyCache.h"

#include <cassert>
#include <regex>
#include <array>

#include "flamegpu/version.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/util/detail/compute_capability.cuh"
#include "flamegpu/util/nvtx.h"

// If MSVC earlier than VS 2019
#if defined(_MSC_VER) && _MSC_VER < 1920
#include <filesystem>
using std::tr2::sys::temp_directory_path;
using std::tr2::sys::exists;
using std::tr2::sys::current_path;
using std::tr2::sys::path;
using std::tr2::sys::directory_iterator;
#else
// VS2019 requires this macro, as building pre c++17 cant use std::filesystem
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
using std::experimental::filesystem::v1::temp_directory_path;
using std::experimental::filesystem::v1::exists;
using std::experimental::filesystem::v1::current_path;
using std::experimental::filesystem::v1::path;
using std::experimental::filesystem::v1::directory_iterator;
#endif

using jitify::detail::hash_combine;
using jitify::detail::hash_larson64;

namespace flamegpu {
namespace util {
namespace detail {

namespace {
/**
 * Returns the tmp dir for storing cache files
 * Defined here to avoid filesystem includes being in header
 */
path getTMP() {
    static path result;
    if (result.empty()) {
        path tmp =  std::getenv("FLAMEGPU_TMP_DIR") ? std::getenv("FLAMEGPU_TMP_DIR") : temp_directory_path();
        // Create the $tmp/flamegpu/jitifycache(/debug) folder hierarchy
        if (!::exists(tmp) && !create_directory(tmp)) {
            THROW exception::InvalidFilePath("Directory '%s' does not exist and cannot be created by JitifyCache.", tmp.generic_string().c_str());
        }
        if (!std::getenv("FLAMEGPU_TMP_DIR")) {
            tmp /= "flamegpu";
            if (!::exists(tmp)) {
                create_directory(tmp);
            }
        }
        tmp /= "jitifycache";
        if (!::exists(tmp)) {
            create_directory(tmp);
        }
#ifdef _DEBUG
        tmp /= "debug";
        if (!::exists(tmp)) {
            create_directory(tmp);
        }
#endif
        result = tmp;
    }
    return result;
}
/**
 * Returns the user-defined include directories
 */
std::vector<path> getIncludeDirs() {
    static std::vector<path> rtn;
    if (rtn.empty()) {
        if (std::getenv("FLAMEGPU_RTC_INCLUDE_DIRS")) {
            const std::string s = std::getenv("FLAMEGPU_RTC_INCLUDE_DIRS");
            // Split the string by ; (windows), : (linux)
#if defined(_MSC_VER)
            std::string delimiter = ";";
#else
            std::string delimiter = ":";
#endif
            size_t start = 0, end = s.find(delimiter);
            std::string token;
            do {
                path p = s.substr(start, end - start);
                if (!p.empty()) {
                    rtn.push_back(p);
                }
                start = end + delimiter.length();
            } while ((end = s.find(delimiter, start))!= std::string::npos);
        } else {
            rtn.push_back(current_path());
        }
    }
    return rtn;
}
std::string loadFile(const path &filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ifstream::binary);
    if (!ifs)
    return "";
    // get length of file
    ifs.seekg(0, ifs.end);
    const std::streamoff length = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::string rtn;
    rtn.resize(length);
    char *buffer = &rtn[0];
    ifs.read(buffer, length);
    ifs.close();
    return rtn;
}

/**
 * Find the cuda include directory.
 * Throws exceptions if it can not be found.
 * @return the path to the CUDA include directory.
 */
std::string getCUDAIncludeDir() {
    // Define an array of environment variables to check in order
    std::array<const std::string, 2> ENV_VARS { "CUDA_PATH", "CUDA_HOME" };
    std::string cuda_include_dir_str = "";
    for (const auto& env_var : ENV_VARS) {
        std::string env_value = std::getenv(env_var.c_str()) ? std::getenv(env_var.c_str()) : "";
        if (!env_value.empty()) {
            path check_path = path(env_value) / "include/";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (exists(check_path)) {
                    cuda_include_dir_str = check_path.string();
                    break;
                }
            } catch (...) { }
            // Throw if the value is not empty, but it does not exist. Outside the try catch excplicityly.
            THROW flamegpu::exception::InvalidFilePath("Error environment variable %s (%s) does not contain a valid CUDA include directory", env_var.c_str(), env_value.c_str());
        }
    }
    // If none of the search enviornmental variables were useful, throw an exception.
    if (cuda_include_dir_str.empty()) {
        THROW exception::InvalidFilePath("Error could not find CUDA include directory. Please specify using the CUDA_PATH environment variable");
    }
    return cuda_include_dir_str;
}

/**
 * Get the FLAME GPU include directory via the environment variables. 
 * @param env_var_used modified to return the name of the environment variable which was used, if any.
 * @return the FLAME GPU 2+ include directory.
 */
std::string getFLAMEGPUIncludeDir(std::string &env_var_used) {
    // Define an array of environment variables to check
    std::array<const std::string, 2> ENV_VARS { "FLAMEGPU_INC_DIR", "FLAMEGPU2_INC_DIR" };
    std::string include_dir_str = "";
    // Iterate the array of environment variables to check for the version header.
    for (const auto& env_var : ENV_VARS) {
        // If the environment variable exists
        std::string env_value = std::getenv(env_var.c_str()) ? std::getenv(env_var.c_str()) : "";
        // If it's a value, check if the path exists, and if any expected files are found.
        if (!env_value.empty()) {
            path check_file = path(env_value) / "flamegpu/flamegpu.h";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (exists(check_file)) {
                    include_dir_str = env_value;
                    env_var_used = env_var;
                    break;
                }
            } catch (...) { }
            // Throw if the value is not empty, but it does not exist. Outside the try catch excplicityly.
            THROW flamegpu::exception::InvalidFilePath("Error environment variable %s (%s) does not contain flamegpu/flamegpu.h. Please correct this environment variable.", env_var.c_str(), env_value.c_str());
        }
    }

    // If no appropriate environmental variables were found, check upwards for N levels (assuming the default filestructure is in use)
    if (include_dir_str.empty()) {
        // Start with the current working directory
        path test_dir(".");
        // Try multiple levels of directory, to see if we can find include/flamegpu/flamegpu.h
        const unsigned int LEVELS = 5;
        for (unsigned int level = 0; level < LEVELS; level++) {
            // If break out the loop if the test_dir directory does not exist.
            if (!exists(test_dir)) {
                break;
            }
            // Check file assuming flamegpu is the root cmake project
            path check_file = test_dir;
            check_file /= "include/flamegpu/flamegpu.h";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (exists(check_file)) {
                    test_dir /= "include";
                    include_dir_str = test_dir.string();
                    break;
                }
            } catch (...) { }
            // Check file assuming a standalone example is the root cmake project
            // We want to see if we can find the build directory
            for (auto& p : directory_iterator(test_dir)) {
                if (is_directory(p)) {
                    check_file = p.path();
                    check_file /= "_deps/flamegpu2-src/include/flamegpu/version.h";
                    // Use try catch to suppress file permission exceptions etc
                    try {
                        if (exists(check_file)) {
                            test_dir = p.path();
                            test_dir /= "_deps/flamegpu2-src/include";
                            include_dir_str = test_dir.string();
                            goto break_flamegpu_inc_dir_loop;  // Break out of nested loop
                        }
                    } catch (...) { }
                }
            }
            // Go up a level for next iteration
            test_dir /= "..";
        }
break_flamegpu_inc_dir_loop:
        // If still not found, throw.
        if (include_dir_str.empty()) {
            // @todo - more appropriate exception?
            THROW flamegpu::exception::InvalidAgentFunc("Error compiling runtime agent function: Unable to automatically determine include directory and FLAMEGPU_INC_DIR environment variable not set");
        }
    }
    return include_dir_str;
}

/**
 * Confirm that include directory version header matches the version of the static library.
 * This only compares up to the pre-release version number. Build metadata is only used for the RTC cache.
 * @param flamegpuIncludeDir path to the flamegpu include directory to check.
 * @return boolean indicator of success.
 */
bool confirmFLAMEGPUHeaderVersion(const std::string flamegpuIncludeDir, const std::string envVariable) {
    static bool header_version_confirmed = false;

    if (!header_version_confirmed) {
        std::string fileHash;
        std::string fileVersionMacro;
        std::string fileVersionPrerelease;
        // Open version.h
        path version_file = path(flamegpuIncludeDir) /= "flamegpu/version.h";
        std::ifstream vFile(version_file);
        if (vFile.is_open()) {
            // Use a regular expression to match the FLAMEGPU_VERSION number macro against lines in the file.
            std::regex macroPattern("^#define FLAMEGPU_VERSION ([0-9]+)$");
            std::regex prereleasePattern("^static constexpr char VERSION_PRERELEASE\\[\\] = \"(.*)\";$");
            std::smatch match;
            std::string line;
            bool extractedMacro = false;
            bool extractedPrerelease = false;
            while (std::getline(vFile, line)) {
                if (std::regex_search(line, match, macroPattern)) {
                    fileVersionMacro = match[1];
                    extractedMacro = true;
                } else if (std::regex_search(line, match, prereleasePattern)) {
                    fileVersionPrerelease = match[1];
                    extractedPrerelease = true;
                }
                if (extractedMacro && extractedPrerelease) {
                    break;
                }
            }
            vFile.close();
            if (!extractedMacro || !extractedPrerelease) {
                THROW exception::VersionMismatch("Could not extract RTC header version information.\n");
            }
        }
        // Confirm that the version matches, else throw an exception.
        if (fileVersionMacro == std::to_string(flamegpu::VERSION) && fileVersionPrerelease == std::string(flamegpu::VERSION_PRERELEASE)) {
            header_version_confirmed = true;
        } else {
            THROW exception::VersionMismatch("RTC header version (%s, %s) does not match version flamegpu library was built with (%s, %s). Set the environment variable %s to the correct include directory.\n",
                fileVersionMacro.c_str(), fileVersionPrerelease.c_str(),
                std::to_string(flamegpu::VERSION).c_str(), flamegpu::VERSION_PRERELEASE,
                envVariable.c_str());
        }
    }
    return header_version_confirmed;
}

}  // namespace

std::mutex JitifyCache::instance_mutex;
std::unique_ptr<KernelInstantiation> JitifyCache::compileKernel(const std::string &func_name, const std::vector<std::string> &template_args, const std::string &kernel_src, const std::string &dynamic_header) {
    NVTX_RANGE("JitifyCache::compileKernel");
    // find and validate the cuda include directory via CUDA_PATH or CUDA_HOME.
    static const std::string cuda_include_dir = getCUDAIncludeDir();
    // find and validate the the flamegpu include directory
    static std::string flamegpu_include_dir_envvar;
    static const std::string flamegpu_include_dir = getFLAMEGPUIncludeDir(flamegpu_include_dir_envvar);
    // verify that the include directory contains the correct headers.
    confirmFLAMEGPUHeaderVersion(flamegpu_include_dir, flamegpu_include_dir_envvar);

     // vector of compiler options for jitify
    std::vector<std::string> options;
    std::vector<std::string> headers;

    // fpgu include directory
    options.push_back(std::string("-I" + std::string(flamegpu_include_dir)));

    // cuda include directory (via CUDA_PATH)
    options.push_back(std::string("-I" + cuda_include_dir));

    // Add user specified include paths
    for (const auto &p : getIncludeDirs())
        options.push_back(std::string("-I" + p.generic_string()));

#ifdef USE_GLM
    // GLM headers increase build time ~5x, so only enable glm if user is using it
    if (kernel_src.find("glm") != std::string::npos) {
        options.push_back(std::string("-I") + GLM_PATH);
        options.push_back(std::string("-DUSE_GLM"));
    }
#endif

    // Set the compilation architecture target if it was successfully detected.
    int currentDeviceIdx = 0;
    cudaError_t status = cudaGetDevice(&currentDeviceIdx);
    if (status == cudaSuccess) {
        int arch = compute_capability::getComputeCapability(currentDeviceIdx);
        options.push_back(std::string("--gpu-architecture=compute_" + std::to_string(arch)));
    }

    // If CUDA is compiled with -G (--device-debug) forward it to the compiler, otherwise forward lineinfo for profiling.
#if defined(__CUDACC_DEBUG__)
    options.push_back("--device-debug");
#else
    options.push_back("--generate-line-info");
#endif

    // If DEBUG is defined, forward it
#if defined(DEBUG)
    options.push_back("-DDEBUG");
#endif

    // If NDEBUG is defined, forward it, this should disable asserts in device code.
#if defined(NDEBUG)
    options.push_back("-DNDEBUG");
#endif

// pass the c++ language dialect. It may be better to explicitly pass this from CMake.
#if defined(__cplusplus) && __cplusplus > 201700L && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 11
    options.push_back("--std=c++17");
#elif defined(__cplusplus) && __cplusplus > 201400L
    options.push_back("--std=c++14");
#endif

    // If SEATBELTS is defined and false, forward it as off, otherwise forward it as on.
#if !defined(SEATBELTS) || SEATBELTS
    options.push_back("--define-macro=SEATBELTS=1");
#else
    options.push_back("--define-macro=SEATBELTS=0");
#endif

    // cuda.h
    std::string include_cuda_h;
    include_cuda_h = "--pre-include=" + cuda_include_dir + "/cuda.h";
    options.push_back(include_cuda_h);

    // get the dynamically generated header from curve rtc
    headers.push_back(dynamic_header);

    // cassert header (to remove remaining warnings) TODO: Ask Jitify to implement safe version of this
    std::string cassert_h = "cassert\n";
    headers.push_back(cassert_h);

    // jitify to create program (with compilation settings)
    try {
        auto program = jitify::experimental::Program(kernel_src, headers, options);
        assert(template_args.size() == 1 || template_args.size() == 3);  // Add this assertion incase template args change
        auto kernel = program.kernel(template_args.size() > 1 ? "flamegpu::agent_function_wrapper" : "flamegpu::agent_function_condition_wrapper");
        return std::make_unique<KernelInstantiation>(kernel, template_args);
    } catch (std::runtime_error const&) {
        // jitify does not have a method for getting compile logs so rely on JITIFY_PRINT_LOG defined in cmake
        THROW exception::InvalidAgentFunc("Error compiling runtime agent function (or function condition) ('%s'): function had compilation errors (see std::cout), "
            "in JitifyCache::buildProgram().",
            func_name.c_str());
    }
}

std::unique_ptr<KernelInstantiation> JitifyCache::loadKernel(const std::string &func_name, const std::vector<std::string> &template_args, const std::string &kernel_src, const std::string &dynamic_header) {
    NVTX_RANGE("JitifyCache::loadKernel");
    std::lock_guard<std::mutex> lock(cache_mutex);
    // Detect current compute capability=
    int currentDeviceIdx = 0;
    cudaError_t status = cudaGetDevice(&currentDeviceIdx);
    const std::string arch = std::to_string((status == cudaSuccess) ? compute_capability::getComputeCapability(currentDeviceIdx) : 0);
    status = cudaRuntimeGetVersion(&currentDeviceIdx);
    const std::string cuda_version = std::to_string((status == cudaSuccess) ? currentDeviceIdx : 0);
    const std::string seatbelts = std::to_string(SEATBELTS);
    // Cat kernel, dynamic header, header version
    const std::string long_reference = kernel_src + dynamic_header;  // Don't need to include rest, they are explicit in short reference/filename
    // Generate short reference string
    // Would prefer to use a proper hash, e.g. md5(reference_string), but that requires extra dependencies
    const std::string short_reference =
        cuda_version + "_" +
        arch + "_" +
        seatbelts + "_" +
        std::string(flamegpu::VERSION_FULL) + "_" +
        // Use jitify hash methods for consistent hashing between OSs
        std::to_string(hash_combine(hash_larson64(kernel_src.c_str()), hash_larson64(dynamic_header.c_str())));
    // Does a copy with the right reference exist in memory?
    if (use_memory_cache) {
        const auto it = cache.find(short_reference);
        if (it != cache.end()) {
            // Check long reference
            if (it->second.long_reference == long_reference) {
                return std::make_unique<KernelInstantiation>(KernelInstantiation::deserialize(it->second.serialised_kernelinst));
            }
        }
    }
    // Does a copy with the right reference exist on disk?
    const path cache_file = getTMP() / short_reference;
    const path reference_file = cache_file.parent_path() / path(cache_file.filename().string() + ".ref");
    if (use_disk_cache && exists(cache_file)) {
        // Load the long reference for the cache file
        const std::string file_long_reference = loadFile(reference_file);
        if (file_long_reference == long_reference) {
            // Load the cache file
            const std::string serialised_kernelinst = loadFile(cache_file);
            if (!serialised_kernelinst.empty()) {
                // Add it to cache for later loads
                cache.emplace(short_reference, CachedProgram{long_reference, serialised_kernelinst});
                // Deserialize and return program
                return std::make_unique<KernelInstantiation>(KernelInstantiation::deserialize(serialised_kernelinst));
            }
        }
    }
    // Kernel has not yet been cached
    {
        // Build kernel
        auto kernelinst = compileKernel(func_name, template_args, kernel_src, dynamic_header);
        // Add it to cache for later loads
        const std::string serialised_kernelinst = use_memory_cache || use_disk_cache ? kernelinst->serialize() : "";
        if (use_memory_cache) {
            cache.emplace(short_reference, CachedProgram{long_reference, serialised_kernelinst});
        }
        // Save it to disk
        if (use_disk_cache) {
            std::ofstream ofs(cache_file, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
            if (ofs) {
                ofs << serialised_kernelinst;
                ofs.close();
            }
            ofs = std::ofstream(reference_file, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
            if (ofs) {
                ofs << long_reference;
                ofs.close();
            }
        }
        return kernelinst;
    }
}
void JitifyCache::useMemoryCache(bool yesno) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    use_memory_cache = yesno;
}
void JitifyCache::useDiskCache(bool yesno) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    use_disk_cache = yesno;
}
bool JitifyCache::useMemoryCache() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return use_memory_cache;
}
bool JitifyCache::useDiskCache() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    return use_disk_cache;
}
void JitifyCache::clearMemoryCache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache.clear();
}
void JitifyCache::clearDiskCache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const path tmp_dir = getTMP();
    for (const auto & entry : directory_iterator(tmp_dir)) {
        if (is_regular_file(entry.path())) {
            remove(entry.path());
        }
    }
}
JitifyCache::JitifyCache()
    : use_memory_cache(true)
#ifndef DISABLE_RTC_DISK_CACHE
    , use_disk_cache(true) { }
#else
    , use_disk_cache(false) { }
#endif
JitifyCache& JitifyCache::getInstance() {
    auto lock = std::unique_lock<std::mutex>(instance_mutex);  // Mutex to protect from two threads triggering the static instantiation concurrently
    static JitifyCache instance;  // Instantiated on first use.
    return instance;
}

}  // namespace detail
}  // namespace util
}  // namespace flamegpu
