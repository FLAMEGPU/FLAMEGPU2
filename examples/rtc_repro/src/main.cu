#include <string>
#include <array>
#include <filesystem>

#ifdef _MSC_VER
#pragma warning(push, 2)
#include "jitify/jitify.hpp"
#pragma warning(pop)
#else
#include "jitify/jitify.hpp"
#endif

#include "flamegpu/detail/compute_capability.cuh"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/detail/SteadyClockTimer.h"

/**
 * Find the cuda include directory.
 * Throws exceptions if it can not be found.
 * @return the path to the CUDA include directory.
 */
std::string getCUDAIncludeDir();
/**
 * Get the FLAME GPU include directory via the environment variables.
 * @param env_var_used modified to return the name of the environment variable which was used, if any.
 * @return the FLAME GPU 2+ include directory.
 */
std::string getFLAMEGPUIncludeDir(std::string& env_var_used);
/**
 * Returns a header normally dynamically generated at runtime
 */
std::string getDynamicHeader();
/**
 * Fill the provided vector with the list of headers we expect to be loaded
 * This enables Jitify to find all the required headers with less NVRTC calls
 *
 * @param headers The vector to fill.
 *
 * @note At current this method has a static list of headers, which is somewhat fragile to changes to our header hierarchy.
 *       In future, Jitify is planning to rework how it processes headers, so this may be unnecessary.
 * @note Libraries such as GLM, which use relative includes internally cannot easily be optimised in this way
 */
void getKnownHeaders(std::vector<std::string>& headers);

int main(void) {
    // Init cuda context (otherwise jitify device api can't find context)
    cudaFree(nullptr);
    
    // find and validate the cuda include directory via CUDA_PATH or CUDA_HOME.
    static const std::string cuda_include_dir = getCUDAIncludeDir();
    // find and validate the the flamegpu include directory
    static std::string flamegpu_include_dir_envvar;
    static const std::string flamegpu_include_dir = getFLAMEGPUIncludeDir(flamegpu_include_dir_envvar);

     // vector of compiler options for jitify
    std::vector<std::string> options;
    std::vector<std::string> headers;

    // fpgu include directory
    options.push_back(std::string("-I" + std::string(flamegpu_include_dir)));

    // cuda include directory (via CUDA_PATH)
    options.push_back(std::string("-I" + cuda_include_dir));

    // Add user specified include paths
    // for (const auto &p : getIncludeDirs())
    //     options.push_back(std::string("-I" + p.generic_string()));

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

    // Set the cuda compuate capability architecture to optimize / generate for, based on the values supported by the current dynamically linked nvrtc and the device in question.
    std::vector<int> nvrtcArchitectures = flamegpu::detail::compute_capability::getNVRTCSupportedComputeCapabilties();
    if (nvrtcArchitectures.size()) {
        int currentDeviceIdx = 0;
        if (cudaSuccess == cudaGetDevice(&currentDeviceIdx)) {
            int arch = flamegpu::detail::compute_capability::getComputeCapability(currentDeviceIdx);
            int maxSupportedArch = flamegpu::detail::compute_capability::selectAppropraiteComputeCapability(arch, nvrtcArchitectures);
            // only set a nvrtc compilation flag if a usable value was found
            if (maxSupportedArch != 0) {
                options.push_back(std::string("--gpu-architecture=compute_" + std::to_string(maxSupportedArch)));
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

    // cuda.h
    std::string include_cuda_h;
    include_cuda_h = "--pre-include=" + cuda_include_dir + "/cuda.h";
    options.push_back(include_cuda_h);

    // get the dynamically generated header from curve rtc
    
    const std::string dynamic_header = getDynamicHeader();
    headers.push_back(dynamic_header);

    // cassert header (to remove remaining warnings) TODO: Ask Jitify to implement safe version of this
    std::string cassert_h = "cassert\n";
    headers.push_back(cassert_h);

    // Add static list of known headers (this greatly improves compilation speed)
    getKnownHeaders(headers);

    // jitify to create program (with compilation settings)
    try {
        const char* kernel_src = R"###(
#include "flamegpu/runtime/DeviceAPI.cuh"
#include "flamegpu/runtime/messaging/MessageNone/MessageNoneDevice.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"
#line 1 "outputdata_impl.cu"
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
}
)###";
        flamegpu::detail::SteadyClockTimer timer;
        timer.start();
        auto program = jitify::experimental::Program(kernel_src, headers, options);
        timer.stop();
        printf("Jitify Program took %fms\n", timer.getElapsedMilliseconds());
        timer = flamegpu::detail::SteadyClockTimer();  // reset
        timer.start();
        auto kernel = program.kernel("flamegpu::agent_function_wrapper");
        timer.stop();
        printf("Jitify Kernel took %fms\n", timer.getElapsedMilliseconds());
        timer = flamegpu::detail::SteadyClockTimer();  // reset
        const std::vector<std::string> template_args = { "outputdata_impl", "flamegpu::MessageNone", "flamegpu::MessageBruteForce" };
        timer.start();
        // Actually compile the kernel
        jitify::experimental::KernelInstantiation(kernel, template_args);
        timer.stop();
        printf("Jitify KernelInstantiation took %fms\n", timer.getElapsedMilliseconds());
        return EXIT_SUCCESS;
    } catch (std::runtime_error const&) {
        // jitify does not have a method for getting compile logs so rely on JITIFY_PRINT_LOG defined in cmake
        throw flamegpu::exception::InvalidAgentFunc("Error compiling runtime agent function: function had compilation errors (see std::cout), "
            "in JitifyCache::buildProgram().");
    }
    return EXIT_FAILURE;
}

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
            }
            catch (...) {}
            // Throw if the value is not empty, but it does not exist. Outside the try catch explicitly.
            throw flamegpu::exception::InvalidFilePath("Error environment variable %s (%s) does not contain a valid CUDA include directory", env_var.c_str(), env_value.c_str());
        }
    }
    // If none of the search environmental variables were useful, throw an exception.
    if (cuda_include_dir_str.empty()) {
        throw flamegpu::exception::InvalidFilePath("Error could not find CUDA include directory. Please specify using the CUDA_PATH environment variable");
    }
    return cuda_include_dir_str;
}

std::string getFLAMEGPUIncludeDir(std::string& env_var_used) {
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
            throw flamegpu::exception::InvalidFilePath("Error environment variable %s (%s) does not contain flamegpu/flamegpu.h. Please correct this environment variable.", env_var.c_str(), env_value.c_str());
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
            throw flamegpu::exception::InvalidAgentFunc("Error compiling runtime agent function: Unable to automatically determine include directory and FLAMEGPU_INC_DIR environment variable not set");
        }
    }
    return include_dir_str;
}

std::string getDynamicHeader() {
    return R"###(dynamic/curve_rtc_dynamic.h
#line 1 "outputdata_impl_curve_rtc_dynamic.h"
#ifndef CURVE_RTC_DYNAMIC_H_
#define CURVE_RTC_DYNAMIC_H_

#include "flamegpu/exception/FLAMEGPUDeviceException.cuh"
#include "flamegpu/detail/type_decode.h"
#include "flamegpu/runtime/detail/curve/Curve.cuh"
#include "flamegpu/util/dstring.h"

namespace flamegpu {

template <unsigned int N, unsigned int I> struct StringCompare {
    __device__ inline static bool strings_equal_loop(const char(&a)[N], const char(&b)[N]) {
        return a[N - I] == b[N - I] && StringCompare<N, I - 1>::strings_equal_loop(a, b);
    }
};

template <unsigned int N> struct StringCompare<N, 1> {
    __device__ inline static bool strings_equal_loop(const char(&a)[N], const char(&b)[N]) {
        return a[0] == b[0];
    }
};

template <unsigned int N>
__device__ bool strings_equal(const char(&a)[N], const char(&b)[N]) {
    return StringCompare<N, N>::strings_equal_loop(a, b);
}

template <unsigned int N, unsigned int M>
__device__ bool strings_equal(const char(&a)[N], const char(&b)[M]) {
    return false;
}

namespace detail {
namespace curve {

/**
 * Dynamically generated version of Curve without hashing
 * Both environment data, and curve variable ptrs are stored in this buffer
 * Order: Env Data, Agent, MessageOut, MessageIn, NewAgent
 * EnvData size must be a multiple of 8 bytes
 */
__constant__  char rtc_env_data_curve[168];


class DeviceCurve {
    public:
    static const int UNKNOWN_VARIABLE = -1;

    typedef int                      Variable;
    typedef unsigned int             VariableHash;
    typedef unsigned int             NamespaceHash;
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable(const char(&name)[N], unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable(const char(&name)[N], unsigned int index);
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable_ldg(const char(&name)[N], unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable_ldg(const char(&name)[N], unsigned int index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable(const char(&name)[M], unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getMessageArrayVariable(const char(&name)[M], unsigned int variable_index, unsigned int array_index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable_ldg(const char(&name)[M], unsigned int variable_index, unsigned int array_index);    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getMessageArrayVariable_ldg(const char(&name)[M], unsigned int variable_index, unsigned int array_index);
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setAgentVariable(const char(&name)[N], T variable, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setMessageVariable(const char(&name)[N], T variable, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setNewAgentVariable(const char(&name)[N], T variable, unsigned int index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setAgentArrayVariable(const char(&name)[M], T variable, unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setMessageArrayVariable(const char(&name)[M], T variable, unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setNewAgentArrayVariable(const char(&name)[M], T variable, unsigned int variable_index, unsigned int array_index);

    __device__ __forceinline__ static bool isAgent(const char* agent_name);
    __device__ __forceinline__ static bool isState(const char* agent_state);
};

template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getAgentVariable(const char (&name)[N], unsigned int index) {
            if (strings_equal(name, "_id")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 56)))[index];
            }
            if (strings_equal(name, "fx")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 64)))[index];
            }
            if (strings_equal(name, "fy")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 72)))[index];
            }
            if (strings_equal(name, "fz")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 80)))[index];
            }
            if (strings_equal(name, "x")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 88)))[index];
            }
            if (strings_equal(name, "y")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 96)))[index];
            }
            if (strings_equal(name, "z")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(detail::type_decode<T>::type_t) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                } else if(detail::type_decode<T>::len_t != 1) {
                    DTHROW("Agent variable '%s' length mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 104)))[index];
            }
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            DTHROW("Agent variable '%s' was not found during getVariable().\n", name);
#endif
            return {};

}
template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getMessageVariable(const char (&name)[N], unsigned int index) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            DTHROW("Message variable '%s' was not found during getVariable().\n", name);
#endif
            return {};

}

template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getAgentVariable_ldg(const char (&name)[N], unsigned int index) {
            if (strings_equal(name, "_id")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 56))) + index);
            }
            if (strings_equal(name, "fx")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 64))) + index);
            }
            if (strings_equal(name, "fy")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 72))) + index);
            }
            if (strings_equal(name, "fz")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 80))) + index);
            }
            if (strings_equal(name, "x")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 88))) + index);
            }
            if (strings_equal(name, "y")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 96))) + index);
            }
            if (strings_equal(name, "z")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
                if(sizeof(T) != 4) {
                    DTHROW("Agent variable '%s' type mismatch during getVariable().\n", name);
                    return {};
                }
#endif
                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 104))) + index);
            }
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            DTHROW("Agent variable '%s' was not found during getVariable().\n", name);
#endif
            return {};

}
template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getMessageVariable_ldg(const char (&name)[N], unsigned int index) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
            DTHROW("Message variable '%s' was not found during getVariable().\n", name);
#endif
            return {};

}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentArrayVariable(const char(&name)[M], unsigned int index, unsigned int array_index) {
    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
           DTHROW("Agent array variable '%s' was not found during getVariable().\n", name);
#endif
           return {};

}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageArrayVariable(const char(&name)[M], unsigned int index, unsigned int array_index) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
           DTHROW("Message array variable '%s' was not found during getVariable().\n", name);
#endif
           return {};

}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentArrayVariable_ldg(const char(&name)[M], unsigned int index, unsigned int array_index) {
    const size_t i = (index * N) + array_index;
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
           DTHROW("Agent array variable '%s' was not found during getVariable().\n", name);
#endif
           return {};

}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageArrayVariable_ldg(const char(&name)[M], unsigned int index, unsigned int array_index) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
           DTHROW("Message array variable '%s' was not found during getVariable().\n", name);
#endif
           return {};

}

template <typename T, unsigned int N>
__device__ __forceinline__ void DeviceCurve::setAgentVariable(const char(&name)[N], T variable, unsigned int index) {
          if (strings_equal(name, "_id")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 56)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "fx")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 64)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "fy")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 72)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "fz")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 80)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "x")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 88)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "y")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 96)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "z")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Agent variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Agent variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 104)))[index] = (T) variable;
              return;
          }
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
          DTHROW("Agent variable '%s' was not found during setVariable().\n", name);
#endif

}
template <typename T, unsigned int N>
__device__ __forceinline__ void DeviceCurve::setMessageVariable(const char(&name)[N], T variable, unsigned int index) {
          if (strings_equal(name, "fx")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 112)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "fy")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 120)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "fz")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 128)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "id")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 136)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "x")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 144)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "y")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 152)))[index] = (T) variable;
              return;
          }
          if (strings_equal(name, "z")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
              if(sizeof(detail::type_decode<T>::type_t) != 4) {
                  DTHROW("Message variable '%s' type mismatch during setVariable().\n", name);
                  return;
              } else if(detail::type_decode<T>::len_t != 1) {
                  DTHROW("Message variable '%s' length mismatch during setVariable().\n", name);
                  return;
              }
#endif
              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 160)))[index] = (T) variable;
              return;
          }
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
          DTHROW("Message variable '%s' was not found during setVariable().\n", name);
#endif

}
template <typename T, unsigned int N>
__device__ __forceinline__ void DeviceCurve::setNewAgentVariable(const char(&name)[N], T variable, unsigned int index) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
          DTHROW("New agent variable '%s' was not found during setVariable().\n", name);
#endif

}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setAgentArrayVariable(const char(&name)[M], T variable, unsigned int index, unsigned int array_index) {
    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
          DTHROW("Agent array variable '%s' was not found during setVariable().\n", name);
#endif
    
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setMessageArrayVariable(const char(&name)[M], T variable, unsigned int index, unsigned int array_index) {
    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
          DTHROW("Message array variable '%s' was not found during setVariable().\n", name);
#endif
    
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setNewAgentArrayVariable(const char(&name)[M], T variable, unsigned int index, unsigned int array_index) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
          DTHROW("New agent array variable '%s' was not found during setVariable().\n", name);
#endif
    
}

__device__ __forceinline__ bool DeviceCurve::isAgent(const char* agent_name) {
    return util::dstrcmp(agent_name, "Boid") == 0;
}
__device__ __forceinline__ bool DeviceCurve::isState(const char* agent_state) {
    return util::dstrcmp(agent_state, "default") == 0;
}

}  // namespace curve 
}  // namespace detail 
}  // namespace flamegpu 

// has to be included after definition of curve namespace
#include "flamegpu/runtime/environment/DeviceEnvironment.cuh"
//#include "flamegpu/runtime/environment/DeviceMacroProperty.cuh"

namespace flamegpu {

template<typename T, unsigned int M>
__device__ __forceinline__ T ReadOnlyDeviceEnvironment::getProperty(const char(&name)[M]) const {
    if (strings_equal(name, "COLLISION_SCALE")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 48));
    };
    if (strings_equal(name, "GLOBAL_SCALE")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 44));
    };
    if (strings_equal(name, "INTERACTION_RADIUS")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 40));
    };
    if (strings_equal(name, "MATCH_SCALE")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 36));
    };
    if (strings_equal(name, "MAX_INITIAL_SPEED")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 32));
    };
    if (strings_equal(name, "MAX_POSITION")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 28));
    };
    if (strings_equal(name, "MIN_INITIAL_SPEED")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 24));
    };
    if (strings_equal(name, "MIN_POSITION")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 20));
    };
    if (strings_equal(name, "POPULATION_TO_GENERATE")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 16));
    };
    if (strings_equal(name, "SEPARATION_RADIUS")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 12));
    };
    if (strings_equal(name, "STEER_SCALE")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 8));
    };
    if (strings_equal(name, "TIME_SCALE")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 4));
    };
    if (strings_equal(name, "_stepCount")) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        if(sizeof(detail::type_decode<T>::type_t) != 4) {
            DTHROW("Environment property '%s' type mismatch.\n", name);
            return {};
        } else if(detail::type_decode<T>::len_t != 1) {
            DTHROW("Environment property '%s' length mismatch.\n", name);
            return {};
        }
#endif
        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::rtc_env_data_curve + 0));
    };
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    DTHROW("Environment property '%s' was not found.\n", name);
#endif
    return  {};

}

template<typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T ReadOnlyDeviceEnvironment::getProperty(const char(&name)[M], const unsigned int index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    DTHROW("Environment array property '%s' was not found.\n", name);
#endif
    return {};

}


template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int N>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W> ReadOnlyDeviceEnvironment::getMacroProperty(const char(&name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    DTHROW("Environment macro property '%s' was not found.\n", name);
    return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(nullptr, nullptr);
#else
    return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(nullptr);
#endif

}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int N>
__device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W> DeviceEnvironment::getMacroProperty(const char(&name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    DTHROW("Environment macro property '%s' was not found.\n", name);
    return DeviceMacroProperty<T, I, J, K, W>(nullptr, nullptr);
#else
    return DeviceMacroProperty<T, I, J, K, W>(nullptr);
#endif

}

}  // namespace flamegpu

#endif  // CURVE_RTC_DYNAMIC_H_

)###";
}

void getKnownHeaders(std::vector<std::string>& headers) {
    // Add known headers from hierarchy
    headers.push_back("algorithm");
    headers.push_back("assert.h");
    headers.push_back("cassert");
    headers.push_back("cfloat");
    headers.push_back("climits");
    headers.push_back("cmath");
    headers.push_back("cstddef");
    headers.push_back("cstdint");
    headers.push_back("cstring");
    headers.push_back("cuda_runtime.h");
    headers.push_back("curand.h");
    headers.push_back("curand_discrete.h");
    headers.push_back("curand_discrete2.h");
    headers.push_back("curand_globals.h");
    headers.push_back("curand_kernel.h");
    headers.push_back("curand_lognormal.h");
    headers.push_back("curand_mrg32k3a.h");
    headers.push_back("curand_mtgp32.h");
    headers.push_back("curand_mtgp32_kernel.h");
    headers.push_back("curand_normal.h");
    headers.push_back("curand_normal_static.h");
    headers.push_back("curand_philox4x32_x.h");
    headers.push_back("curand_poisson.h");
    headers.push_back("curand_precalc.h");
    headers.push_back("curand_uniform.h");
    headers.push_back("device_launch_parameters.h");
    // headers.push_back("dynamic/curve_rtc_dynamic.h");  // This is already included with source, having this makes a vague compile err
    headers.push_back("flamegpu/defines.h");
    headers.push_back("flamegpu/exception/FLAMEGPUDeviceException.cuh");
    headers.push_back("flamegpu/exception/FLAMEGPUDeviceException_device.cuh");
    headers.push_back("flamegpu/simulation/detail/CUDAScanCompaction.h");
    headers.push_back("flamegpu/runtime/AgentFunction.cuh");
    headers.push_back("flamegpu/runtime/AgentFunctionCondition.cuh");
    headers.push_back("flamegpu/runtime/AgentFunctionCondition_shim.cuh");
    headers.push_back("flamegpu/runtime/AgentFunction_shim.cuh");
    headers.push_back("flamegpu/runtime/DeviceAPI.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageArray.h");
    headers.push_back("flamegpu/runtime/messaging/MessageArray/MessageArrayDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageArray2D.h");
    headers.push_back("flamegpu/runtime/messaging/MessageArray2D/MessageArray2DDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageArray3D.h");
    headers.push_back("flamegpu/runtime/messaging/MessageArray3D/MessageArray3DDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageBruteForce.h");
    headers.push_back("flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageBucket.h");
    headers.push_back("flamegpu/runtime/messaging/MessageBucket/MessageBucketDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageSpatial2D.h");
    headers.push_back("flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageSpatial3D.h");
    headers.push_back("flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DDevice.cuh");
    headers.push_back("flamegpu/runtime/messaging/MessageNone.h");
    headers.push_back("flamegpu/runtime/random/AgentRandom.cuh");
    headers.push_back("flamegpu/runtime/environment/DeviceEnvironment.cuh");
    headers.push_back("flamegpu/runtime/environment/DeviceMacroProperty.cuh");
    headers.push_back("flamegpu/detail/StaticAssert.h");
    headers.push_back("flamegpu/detail/type_decode.h");
    // headers.push_back("jitify_preinclude.h");  // I think Jitify adds this itself
    headers.push_back("limits");
    headers.push_back("limits.h");
    headers.push_back("math.h");
    headers.push_back("memory.h");
    headers.push_back("stddef.h");
    headers.push_back("stdint.h");
    headers.push_back("stdio.h");
    headers.push_back("stdlib.h");
    headers.push_back("string");
    headers.push_back("string.h");
    headers.push_back("time.h");
    headers.push_back("type_traits");
}