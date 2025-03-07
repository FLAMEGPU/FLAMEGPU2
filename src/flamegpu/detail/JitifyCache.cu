#include "flamegpu/detail/JitifyCache.h"

#include <nvrtc.h>

#include <cassert>
#include <regex>
#include <array>
#include <filesystem>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cstdio>

#include "jitify/jitify2.hpp"

#include "flamegpu/version.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/detail/compute_capability.cuh"
#include "flamegpu/util/nvtx.h"


namespace flamegpu {
namespace detail {

namespace {
/**
 * Returns the tmp dir for storing cache files
 * Defined here to avoid filesystem includes being in header
 */
std::filesystem::path getTMP() {
    static std::filesystem::path result;
    if (result.empty()) {
        std::filesystem::path tmp =  std::getenv("FLAMEGPU_TMP_DIR") ? std::getenv("FLAMEGPU_TMP_DIR") : std::filesystem::temp_directory_path();
        // Create the $tmp/flamegpu/jitifycache(/debug) folder hierarchy
        if (!std::filesystem::exists(tmp) && !std::filesystem::create_directories(tmp)) {
            THROW exception::InvalidFilePath("Directory '%s' does not exist and cannot be created by JitifyCache.", tmp.generic_string().c_str());
        }
        if (!std::getenv("FLAMEGPU_TMP_DIR")) {
            tmp /= "flamegpu";
            if (!std::filesystem::exists(tmp)) {
                std::filesystem::create_directories(tmp);
            }
        }
        tmp /= "jitifycache";
        if (!std::filesystem::exists(tmp)) {
            std::filesystem::create_directories(tmp);
        }
#ifdef _DEBUG
        tmp /= "debug";
        if (!std::filesystem::exists(tmp)) {
            std::filesystem::create_directories(tmp);
        }
#endif
        result = tmp;
    }
    return result;
}
/**
 * Returns the user-defined include directories
 */
std::vector<std::filesystem::path> getIncludeDirs() {
    static std::vector<std::filesystem::path> rtn;
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
                std::filesystem::path p = s.substr(start, end - start);
                if (!p.empty()) {
                    rtn.push_back(p);
                }
                start = end + delimiter.length();
            } while ((end = s.find(delimiter, start))!= std::string::npos);
        } else {
            rtn.push_back(std::filesystem::current_path());
        }
    }
    return rtn;
}

std::string loadFile(const std::filesystem::path &filepath) {
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
            std::filesystem::path check_path = std::filesystem::path(env_value) / "include/";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (std::filesystem::exists(check_path)) {
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
            std::filesystem::path check_file = std::filesystem::path(env_value) / "flamegpu/flamegpu.h";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (std::filesystem::exists(check_file)) {
                    include_dir_str = env_value;
                    env_var_used = env_var;
                    break;
                }
            } catch (...) { }
            // Throw if the value is not empty, but it does not exist. Outside the try catch explicitly.
            THROW flamegpu::exception::InvalidFilePath("Error environment variable %s (%s) does not contain flamegpu/flamegpu.h. Please correct this environment variable.", env_var.c_str(), env_value.c_str());
        }
    }

    // If no appropriate environmental variables were found, check upwards for N levels (assuming the default file structure is in use)
    if (include_dir_str.empty()) {
        // Start with the current working directory
        std::filesystem::path test_dir(".");
        // Try multiple levels of directory, to see if we can find include/flamegpu/flamegpu.h
        const unsigned int LEVELS = 5;
        for (unsigned int level = 0; level < LEVELS; level++) {
            // If break out the loop if the test_dir directory does not exist.
            if (!std::filesystem::exists(test_dir)) {
                break;
            }
            // Check file assuming flamegpu is the root cmake project
            std::filesystem::path check_file = test_dir;
            check_file /= "include/flamegpu/flamegpu.h";
            // Use try catch to suppress file permission exceptions etc
            try {
                if (std::filesystem::exists(check_file)) {
                    test_dir /= "include";
                    include_dir_str = test_dir.string();
                    break;
                }
            } catch (...) { }
            // Check file assuming a standalone example is the root cmake project
            // We want to see if we can find the build directory
            for (auto& p : std::filesystem::directory_iterator(test_dir)) {
                if (std::filesystem::is_directory(p)) {
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
    return std::filesystem::absolute(include_dir_str).generic_string();
}

#ifdef FLAMEGPU_USE_GLM
/**
 * Get the GLM include directory via the environment variables.
 * @return the GLM include directory.
 */
std::string getGLMIncludeDir() {
    const std::string env_var = "FLAMEGPU_GLM_INC_DIR";
    const std::string test_file = "glm/glm.hpp";
    // Check the environment variable to see whether glm/glm.hpp exists
    {
        // If the environment variable exists
        std::string env_value = std::getenv(env_var.c_str()) ? std::getenv(env_var.c_str()) : "";
        // If it's a value, check if the path exists, and if any expected files are found.
        if (!env_value.empty()) {
            std::filesystem::path check_file = std::filesystem::path(env_value) / test_file;
            // Use try catch to suppress file permission exceptions etc
            try {
                if (std::filesystem::exists(check_file)) {
                    return env_value;
                }
            }
            catch (...) {}
            // Throw if the value is not empty, but it does not exist. Outside the try catch explicitly.
            THROW flamegpu::exception::InvalidFilePath("Error environment variable %s (%s) does not contain %s. Please correct this environment variable.", env_var.c_str(), env_value.c_str(), test_file.c_str());
        }
    }

    // If no appropriate environmental variables were found, check the compile time path to GLM
    std::filesystem::path check_file = std::filesystem::path(FLAMEGPU_GLM_PATH) / test_file;
    // Use try catch to suppress file permission exceptions etc
    try {
        if (std::filesystem::exists(check_file)) {
            return FLAMEGPU_GLM_PATH;
        }
    }
    catch (...) {}
    // Throw if header wasn't found. Outside the try catch explicitly.
    THROW flamegpu::exception::InvalidAgentFunc("Error compiling runtime agent function: Unable to automatically determine location of GLM include directory and %s environment variable not set", env_var.c_str());
}
#endif

/**
 * Confirm that include directory version header matches the version of the static library.
 * This only compares up to the pre-release version number. Build metadata is only used for the RTC cache.
 * @param flamegpuIncludeDir path to the flamegpu include directory to check.
 * @return boolean indicator of success.
 */
bool confirmFLAMEGPUHeaderVersion(const std::string &flamegpuIncludeDir, const std::string &envVariable) {
    static bool header_version_confirmed = false;

    if (!header_version_confirmed) {
        std::string fileHash;
        std::string fileVersionMacro;
        std::string fileVersionPrerelease;
        // Open version.h
        std::filesystem::path version_file = std::filesystem::path(flamegpuIncludeDir) /= "flamegpu/version.h";
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
std::unique_ptr<jitify2::LinkedProgramData> JitifyCache::buildProgram(
    const std::string &func_name,
    const std::vector<std::string> &template_args,
    const std::string &kernel_src,
    const std::string &dynamic_header,
    const std::string &name_expression) {
    flamegpu::util::nvtx::Range range{"JitifyCache::preprocessKernel"};
    // find and validate the cuda include directory via CUDA_PATH or CUDA_HOME.
    static const std::string cuda_include_dir = getCUDAIncludeDir();
    // find and validate the the flamegpu include directory
    static std::string flamegpu_include_dir_envvar;
    static const std::string flamegpu_include_dir = getFLAMEGPUIncludeDir(flamegpu_include_dir_envvar);
    // verify that the include directory contains the correct headers.
    confirmFLAMEGPUHeaderVersion(flamegpu_include_dir, flamegpu_include_dir_envvar);

     // vector of compiler options for jitify
    std::vector<std::string> options;
    std::unordered_map<std::string, std::string> headers;

    // fpgu include directory
    options.push_back(std::string("-I" + std::string(flamegpu_include_dir)));

    // cuda include directory (via CUDA_PATH)
    options.push_back(std::string("-I" + cuda_include_dir));

    // Add user specified include paths
    for (const auto &p : getIncludeDirs())
        options.push_back(std::string("-I" + p.generic_string()));

#ifdef FLAMEGPU_USE_GLM
    // GLM headers increase build time ~5x, so only enable glm if user is using it
    if (kernel_src.find("glm") != std::string::npos) {
        static const std::string glm_include_dir = getGLMIncludeDir();
        options.push_back(std::string("-I") + glm_include_dir);
        options.push_back(std::string("-DFLAMEGPU_USE_GLM"));
    }
#endif

    // Forward the curand Engine request
#if defined(FLAMEGPU_CURAND_MRG32k3a)
    options.push_back(std::string("-DFLAMEGPU_CURAND_MRG32k3a"));
#elif defined(FLAMEGPU_CURAND_Philox4_32_10)
    options.push_back(std::string("-DFLAMEGPU_CURAND_Philox4_32_10"));
#elif defined(FLAMEGPU_CURAND_XORWOW)
    options.push_back(std::string("-DFLAMEGPU_CURAND_XORWOW"));
#endif

    // Set the cuda compuate capability architecture to optimize / generate for, based on the values supported by the current dynamiclaly linked nvrtc and the device in question.
    std::vector<int> nvrtcArchitectures = detail::compute_capability::getNVRTCSupportedComputeCapabilties();
    if (nvrtcArchitectures.size()) {
        int currentDeviceIdx = 0;
        if (cudaSuccess == cudaGetDevice(&currentDeviceIdx)) {
            int arch = compute_capability::getComputeCapability(currentDeviceIdx);
            int maxSupportedArch = compute_capability::selectAppropraiteComputeCapability(arch, nvrtcArchitectures);
            // only set a nvrtc compilation flag if a usable value was found
            if (maxSupportedArch != 0) {
                options.push_back(std::string("--gpu-architecture=compute_" + std::to_string(maxSupportedArch)));
            } else {
                // This branch should never be taken
                // Rather than throwing an exception which users cannot catch and reover from, assert instead. This will just result in not targetting a specific arch.
                assert(false);
            }
        }
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
#endif

    // If FLAMEGPU_SEATBELTS is defined and false, forward it as off, otherwise forward it as on.
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    options.push_back("--define-macro=FLAMEGPU_SEATBELTS=1");
#else
    options.push_back("--define-macro=FLAMEGPU_SEATBELTS=0");
#endif

    // get the dynamically generated header from curve rtc
    headers.emplace("dynamic/curve_rtc_dynamic.h", dynamic_header);

    // jitify to create program (with compilation settings)
    const std::string program_name = func_name + "_program";  // Does this name actually matter?
    jitify2::PreprocessedProgram program = jitify2::Program(program_name, kernel_src, headers)->preprocess(options);
    if (!program.ok()) {
        const jitify2::ErrorMsg& compile_error = program.error();
        fprintf(stderr, "Failed to load program for agent function (condition) '%s', log:\n%s",
            func_name.c_str(), compile_error.c_str());
        THROW exception::InvalidAgentFunc("Error loading agent function (or function condition) ('%s'): function had compilation errors:\n%s",
            func_name.c_str(), compile_error.c_str());
    }
    // Compile
    jitify2::CompiledProgram compiled_program = program->compile({ name_expression });
    if (!compiled_program.ok()) {
        const jitify2::ErrorMsg& compile_error = compiled_program.error();
        fprintf(stderr, "Failed to compile agent function (condition) '%s', log:\n%s",
            func_name.c_str(), compile_error.c_str());
        THROW exception::InvalidAgentFunc("Error loading agent function (or function condition) ('%s'): function had compilation errors:\n%s",
            func_name.c_str(), compile_error.c_str());
    }
    // Link
    jitify2::LinkedProgram linked_program = compiled_program->link();
    if (!linked_program.ok()) {
        const jitify2::ErrorMsg& link_error = linked_program.error();
        fprintf(stderr, "Failed to link agent function (condition) '%s', log:\n%s",
            func_name.c_str(), link_error.c_str());
        THROW exception::InvalidAgentFunc("Error loading agent function (or function condition) ('%s'): function had link errors:\n%s",
            func_name.c_str(), link_error.c_str());
    }
    return std::make_unique<jitify2::LinkedProgramData>(linked_program.value());
}
std::unique_ptr<jitify2::KernelData> JitifyCache::loadKernel(const std::string &func_name, const std::vector<std::string> &template_args, const std::string &kernel_src, const std::string &dynamic_header) {
    flamegpu::util::nvtx::Range range{"JitifyCache::loadKernel"};
    std::lock_guard<std::mutex> lock(cache_mutex);
    // Detect current compute capability=
    int currentDeviceIdx = 0;
    cudaError_t status = cudaGetDevice(&currentDeviceIdx);
    const std::string arch = std::to_string((status == cudaSuccess) ? compute_capability::getComputeCapability(currentDeviceIdx) : 0);
    status = cudaRuntimeGetVersion(&currentDeviceIdx);
    const std::string cuda_version = std::to_string((status == cudaSuccess) ? currentDeviceIdx : 0);
    const std::string seatbelts = std::to_string(FLAMEGPU_SEATBELTS);
    // Cat kernel, dynamic header, header version
    const std::string long_reference = kernel_src + dynamic_header;  // Don't need to include rest, they are explicit in short reference/filename
    // Generate short reference string
    // Would prefer to use a proper hash, e.g. md5(reference_string), but that requires extra dependencies
    const std::string short_reference =
        cuda_version + "_" +
        arch + "_" +
        seatbelts + "_" +
        std::string(flamegpu::VERSION_FULL) + "_" +
#ifdef FLAMEGPU_USE_GLM
        "glm_" +
#endif
#if defined(FLAMEGPU_CURAND_MRG32k3a)
        "MRG_" +
#elif defined(FLAMEGPU_CURAND_Philox4_32_10)
        "PHILOX_" +
#elif defined(FLAMEGPU_CURAND_XORWOW)
        "XORWOW_" +
#endif
        // Use jitify hash methods for consistent hashing between OSs
        jitify2::detail::sha256(kernel_src + dynamic_header);
    std::unique_ptr<jitify2::LinkedProgramData> linked_program;
    // Does a copy with the right reference exist in memory?
    if (use_memory_cache) {
        const auto it = cache.find(short_reference);
        if (it != cache.end()) {
            // Check long reference
            if (it->second.long_reference == long_reference) {
                // Deserialize and return program
                jitify2::LinkedProgram prog = jitify2::LinkedProgram::deserialize(it->second.serialised_program);
                if (prog.ok()) {
                    linked_program = std::make_unique<jitify2::LinkedProgramData>(prog.value());
                }
                // Fail silently and try to build code
            }
        }
    }
    // Does a copy with the right reference exist on disk?
    const std::filesystem::path cache_file = getTMP() / short_reference;
    const std::filesystem::path reference_file = cache_file.parent_path() / std::filesystem::path(cache_file.filename().string() + ".ref");
    if (!linked_program && use_disk_cache && std::filesystem::exists(cache_file)) {
        // Load the long reference for the cache file
        const std::string file_long_reference = loadFile(reference_file);
        if (file_long_reference == long_reference) {
            // Load the cache file
            const std::string serialised_kernelinst = loadFile(cache_file);
            if (!serialised_kernelinst.empty()) {
                // Add it to cache for later loads
                cache.emplace(short_reference, CachedProgram{long_reference, serialised_kernelinst});
                // Deserialize and return program
                jitify2::LinkedProgram prog = jitify2::LinkedProgram::deserialize(serialised_kernelinst);
                if (prog.ok()) {
                    linked_program = std::make_unique<jitify2::LinkedProgramData>(prog.value());
                }
                // Fail silently and try to build code
            }
        }
    }
    // Build the name of the template configuration to be instantiated
    std::stringstream name_expression;
    if (template_args.size() == 1) {
        name_expression << "flamegpu::agent_function_condition_wrapper<";
        name_expression << template_args[0];
        name_expression << ">";
    } else if (template_args.size() == 3) {
        name_expression << "flamegpu::agent_function_wrapper<";
        name_expression << template_args[0] << "," << template_args[1] << "," << template_args[2];
        name_expression << ">";
    } else {
        THROW exception::UnknownInternalError("Unexpected AgentFunction template arg count!");
    }
    // Kernel has not yet been cached
    if (!linked_program) {
        // Build kernel
        linked_program = buildProgram(func_name, template_args, kernel_src, dynamic_header, name_expression.str());
        // Add it to cache for later loads
        const std::string serialised_program = use_memory_cache || use_disk_cache ? linked_program->serialize() : "";
        if (use_memory_cache) {
            cache.emplace(short_reference, CachedProgram{long_reference, serialised_program });
        }
        // Save it to disk
        if (use_disk_cache) {
            std::ofstream ofs(cache_file, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
            if (ofs) {
                ofs << serialised_program;
                ofs.close();
            }
            ofs = std::ofstream(reference_file, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
            if (ofs) {
                ofs << long_reference;
                ofs.close();
            }
        }
    }
    jitify2::LoadedProgram loaded_program = linked_program->load();
    if (!loaded_program.ok()) {
        const jitify2::ErrorMsg& compile_error = loaded_program.error();
        fprintf(stderr, "Failed to load program for agent function (condition) '%s' into memory, log:\n%s",
            func_name.c_str(), compile_error.c_str());
        THROW exception::InvalidAgentFunc("Error loading agent function (or function condition) ('%s'): function had errors (see std::out), "
            "in JitifyCache::loadKernel().",
            func_name.c_str());
    }
    jitify2::Kernel loaded_kernel = loaded_program->get_kernel(name_expression.str());
    if (loaded_kernel.ok()) {
        return std::make_unique<jitify2::KernelData>(loaded_kernel.value());
    }
    const jitify2::ErrorMsg& compile_error = loaded_kernel.error();
    fprintf(stderr, "Failed to compile and link agent function (condition) '%s', log:\n%s",
        func_name.c_str(), compile_error.c_str());
    THROW exception::InvalidAgentFunc("Error compiling runtime agent function (or function condition) ('%s'): function had compilation errors (see std::cout), "
        "in JitifyCache::loadKernel().",
        func_name.c_str());
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
    const std::filesystem::path tmp_dir = getTMP();
    for (const auto & entry : std::filesystem::directory_iterator(tmp_dir)) {
        if (std::filesystem::is_regular_file(entry.path())) {
            remove(entry.path());
        }
    }
}
JitifyCache::JitifyCache()
    : use_memory_cache(true)
#ifndef FLAMEGPU_DISABLE_RTC_DISK_CACHE
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
}  // namespace flamegpu
