#ifndef INCLUDE_FLAMEGPU_UTIL_NVTX_H_
#define INCLUDE_FLAMEGPU_UTIL_NVTX_H_

#include <cstdint>

// If NVTX is enabled, include the appropriate header and define some internal types
#if defined(FLAMEGPU_USE_NVTX)
    #if defined(FLAMEGPU_USE_CUDA)
        #include "nvtx3/nvToolsExt.h"
    #endif  // defined(FLAMEGPU_USE_CUDA)
    #if defined(FLAMEGPU_USE_HIP)
        #include <rocprofiler-sdk-roctx/roctx.h>
    #endif  // defined(FLAMEGPU_USE_HIP)
#endif  // defined(FLAMEGPU_USE_NVTX)

namespace flamegpu {
namespace util {

/**
 * Utility namespace for handling of NVTX profiling markers/ranges, uses if constexpr to avoid runtime cost when disabled
 *
 * Macro `FLAMEGPU_USE_NVTX` is defined via CMake to set the member constexpr
 */
namespace nvtx {

/**
 * Colour palette for NVTX markers in ARGB format.
 * From colour brewer qualitative 8-class Dark2
 */
static constexpr uint32_t palette[] = {0xff1b9e77, 0xffd95f02, 0xff7570b3, 0xffe7298a, 0xff66a61e, 0xffe6ab02, 0xffa6761d, 0xff666666};

/**
 * The number of colours in the nvtx marker palette
 */
static constexpr uint32_t colourCount = sizeof(palette) / sizeof(uint32_t);

/**
 * Namespace-scoped constant expression indicating if NVTX support is enabled or not based on the FLAMEGPU_USE_NVTX macro.
 */
#if defined(FLAMEGPU_USE_NVTX)
static constexpr bool ENABLED = true;
#else
static constexpr bool ENABLED = false;
#endif

/**
 * Method to push an NVTX marker for improved profiling, if NVTX is enabled
 * @param label label for the NVTX marker
 * @note The number of pushes must match the number of pops.
 */
inline void push(const char * label) {
    // Only do anything if nvtx is enabled, but also need to macro guard things from the  guarded headers
    #if defined(FLAMEGPU_USE_NVTX)
        if constexpr (ENABLED) {
        #if defined(FLAMEGPU_USE_CUDA)
            // Static variable to track the next colour to be used with auto rotation.
            static uint32_t nextColourIdx = 0;

            // Get the wrapped colour index
            uint32_t colourIdx = nextColourIdx % colourCount;

            // Build/populate the struct of nvtx event attributes
            nvtxEventAttributes_t eventAttrib = {0};
            // Generic values
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib.colorType = NVTX_COLOR_ARGB;
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;

            // Selected colour and string
            eventAttrib.color = palette[colourIdx];
            eventAttrib.message.ascii = label;

            // Push the custom event.
            nvtxRangePushEx(&eventAttrib);

            // Increment the counter tracking the next colour to use.
            nextColourIdx = colourIdx + 1;
        #else  // defined(FLAMEGPU_USE_HIP)
            // roctx does not include an equivalent to nvtxRangePushEx, can only use the naive version
            roctxRangePush(label);
        #endif
        }
    #endif  // defined(FLAMEGPU_USE_NVTX)
}

/**
 * Method to pop an NVTX marker for improved profiling, if NVTX is enabled at configuration time.
 * @note the number of pops must match the number of pushes.
 */
inline void pop() {
    // Only do anything if nvtx is enabled
    #if defined(FLAMEGPU_USE_NVTX)
        if constexpr (ENABLED) {
        #if defined(FLAMEGPU_USE_CUDA)
            nvtxRangePop();
        #else  // defined(FLAMEGPU_USE_HIP)
            roctxRangePop();
        #endif
        }
    #endif
}

/**
 * Scope-based NVTX ranges.
 * Push at construction, Pop at destruction.
 */
class Range {
 public:
    /**
    * Constructor which pushes an NVTX marker onto the stack with the specified label
    * @param label the label for the nvtx range.
    */
    explicit Range(const char *label) {
        if constexpr (nvtx::ENABLED) {
            nvtx::push(label);
        }
    }
    /**
     *  Destructor which pops a marker off the nvtx stack.
     */
    ~Range() {
        if constexpr (nvtx::ENABLED) {
            nvtx::pop();
        }
    }
};

}  // namespace nvtx
}  // namespace util
}  // namespace flamegpu


#endif  // INCLUDE_FLAMEGPU_UTIL_NVTX_H_
