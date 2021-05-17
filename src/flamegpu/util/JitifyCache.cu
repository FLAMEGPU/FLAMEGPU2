#include "flamegpu/util/JitifyCache.h"

#include <cassert>

#include "flamegpu/version.h"
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/util/compute_capability.cuh"
#include "flamegpu/util/nvtx.h"

// If MSVC earlier than VS 2019
#if defined(_MSC_VER) && _MSC_VER < 1920
#include <filesystem>
using std::tr2::sys::temp_directory_path;
using std::tr2::sys::exists;
using std::tr2::sys::path;
using std::tr2::sys::directory_iterator;
#else
// VS2019 requires this macro, as building pre c++17 cant use std::filesystem
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
using std::experimental::filesystem::v1::temp_directory_path;
using std::experimental::filesystem::v1::exists;
using std::experimental::filesystem::v1::path;
using std::experimental::filesystem::v1::directory_iterator;
#endif

using jitify::detail::hash_combine;
using jitify::detail::hash_larson64;

namespace flamegpu {
namespace util {

namespace {
/**
 * Returns the tmp dir for storing cache files
 * Defined here to avoid filesystem includes being in header
 */
path getTMP() {
    static path result;
    if (result.empty()) {
        path tmp =  std::getenv("FLAMEGPU2_TMP_DIR") ? std::getenv("FLAMEGPU2_TMP_DIR") : temp_directory_path();
        // Create the $tmp/fgpu2/jitifycache(/debug) folder hierarchy
        if (!::exists(tmp) && !create_directory(tmp)) {
            THROW InvalidFilePath("Directory '%s' does not exist and cannot be created by JitifyCache.", tmp.generic_string().c_str());
        }
        if (!std::getenv("FLAMEGPU2_TMP_DIR")) {
            tmp /= "fgpu2";
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
}  // namespace

std::mutex JitifyCache::instance_mutex;
std::unique_ptr<KernelInstantiation> JitifyCache::compileKernel(const std::string &func_name, const std::vector<std::string> &template_args, const std::string &kernel_src, const std::string &dynamic_header) {
    NVTX_RANGE("JitifyCache::compileKernel");
    // Init runtime compilation constants
    static std::string env_inc_fgp2 = std::getenv("FLAMEGPU2_INC_DIR") ? std::getenv("FLAMEGPU2_INC_DIR") : "";
    static bool header_version_confirmed = false;
    static std::string env_cuda_path = std::getenv("CUDA_PATH") ? std::getenv("CUDA_PATH") : "";
    if (env_inc_fgp2.empty()) {
        // Start with the current working directory
        path test_include(".");
        // Try 5 levels of directory, to see if we can find flame_api.h
        for (int i = 0; i < 5; ++i) {
            // If break out the loop if the test_include directory does not exist.
            if (!exists(test_include)) {
                break;
            }
            // Check file assuming flamegpu2 is the root cmake project
            path check_file = test_include;
            check_file/= "include/flamegpu/version.h";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (exists(check_file)) {
                    test_include /= "include";
                    env_inc_fgp2 = test_include.string();
                    break;
                }
            } catch (...) { }
            // Check file assuming a standalone example is the root cmake project
            // We want to see if we can find the build directory
            for (auto& p : directory_iterator(test_include)) {
                if (is_directory(p)) {
                    check_file = p.path();
                    check_file /= "_deps/flamegpu2-src/include/flamegpu/version.h";
                    // Use try catch to suppress file permission exceptions etc
                    try {
                        if (exists(check_file)) {
                            test_include = p.path();
                            test_include /= "_deps/flamegpu2-src/include";
                            env_inc_fgp2 = test_include.string();
                            goto break_fgpu2_inc_dir_loop;  // Break out of nested loop
                        }
                    } catch (...) { }
                }
            }
            // Go up a level for next iteration
            test_include/= "..";
        }
break_fgpu2_inc_dir_loop:
        if (env_inc_fgp2.empty()) {
            THROW InvalidAgentFunc("Error compiling runtime agent function: Unable to automatically determine include directory and FLAMEGPU2_INC_DIR environment variable does not exist, "
                "in JitifyCache::compileKernel().");
        }
    }
    if (!header_version_confirmed) {
        std::string fileHash;
        // Open version.h
        path version_file = env_inc_fgp2;
        version_file/= "flamegpu/version.h";
        std::ifstream vFile(version_file);
        if (vFile.is_open()) {
            // Read the first line
            std::string line;
            if (getline(vFile, line)) {
                // If characters 3-onwards match programatic hash we have success, else fail
                fileHash = line.substr(3, std::string::npos);
            }
            vFile.close();
        }
        if (fileHash == detail::getCommitHash()) {
            header_version_confirmed = true;
        } else {
            THROW VersionMismatch("RTC header version (%s) does not match version flamegpu2 library was built with (%s). Set the environment variable FLAMEGPU2_INC_DIR to the correct include directory.\n",
                fileHash.c_str(), detail::getCommitHash().c_str());
        }
    }
    if (env_cuda_path.empty()) {
        THROW InvalidAgentFunc("Error compiling runtime agent function: CUDA_PATH environment variable does not exist, "
            "in CUDAAgent::compileKernel().");
    }
    // If the last char is a / or \, remove it. Only removes a single slash.
    if ((env_cuda_path.back() == '/' || env_cuda_path.back() == '\\')) {
        env_cuda_path.pop_back();
    }

     // vector of compiler options for jitify
    std::vector<std::string> options;
    std::vector<std::string> headers;

    // fpgu include directory
    options.push_back(std::string("-I" + std::string(env_inc_fgp2)));

    // cuda include directory (via CUDA_PATH)
    options.push_back(std::string("-I" + env_cuda_path + "/include"));

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
    include_cuda_h = "--pre-include=" + env_cuda_path + "/include/cuda.h";
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
        THROW InvalidAgentFunc("Error compiling runtime agent function (or function condition) ('%s'): function had compilation errors (see std::cout), "
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
        detail::getCommitHash() + "_" +
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
    const path reference_file = path(cache_file).replace_extension(".ref");
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

}  // namespace util
}  // namespace flamegpu
