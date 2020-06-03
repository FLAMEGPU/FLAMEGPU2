#ifndef INCLUDE_FLAMEGPU_UTIL_NVTX_H_
#define INCLUDE_FLAMEGPU_UTIL_NVTX_H_

/**
 * Utility namespace for handling of NVTX profiling markers/ranges, wrapped in macros to avoid performance impact if not enabled.
 * 
 * Macro `USE_NVTX` must be defined to be enabled.
 * Use NVTX_PUSH, NVTX_POP, NVTX_RANGE macros to use.
 */

// If NVTX is enabled, include header, defined namespace / class and macros.
#if defined(USE_NVTX)
    // Include the appropriate header if enabled
    #if USE_NVTX >= 3
        #include "nvtx3/nvToolsExt.h"
    #else
        #include "nvToolsExt.h"
    #endif
#endif

/* @todo - Make these macros testable.
   If USE_NVTX is enabled, store static counts of push/pop/range's
   Make accessors to enable testing the number of counts is as expected
   Could also include this in a device shutdown method, to report if there is a mismatch of push/pop and therefore an NVTX error.
*/

namespace util {
namespace nvtx {

/**
 * Colour palette for NVTX markers in ARGB format.
 * From colour brewer qualitative 8-class Dark2
 */
const uint32_t palette[] = {0xff1b9e77, 0xffd95f02, 0xff7570b3, 0xffe7298a, 0xff66a61e, 0xffe6ab02, 0xffa6761d, 0xff666666};

/**
 * The number of colours in the nvtx marker palette
 */
const uint32_t colourCount = sizeof(palette) / sizeof(uint32_t);

/**
 * Method to push an NVTX marker for improved profiling, if NVTX is defined
 * @param label label for the NVTX marker
 * @note The number of pushes must match the number of pops.
 * @see NVTX_PUSH to use with minimal performance impact
 */
#if defined(USE_NVTX)
inline void push(const char * label) {
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
}
#else
inline void push(const char *) {
}
#endif

/**
 * Method to pop an NVTX marker for improved profiling, if NVTX is defined.
 * @note the number of pops must match the number of pushes.
 * @see NVTX_POP to use with minimal performance impact
 */
inline void pop() {
    #if defined(USE_NVTX)
        nvtxRangePop();
    #endif
}

/**
 * Scope-based NVTX ranges. 
 * Push at construction, Pop at destruction.
 */
class NVTXRange {
 public:
    /**
    * Constuctor which pushes an NVTX marker onto the stack with the specified label
    * @param label the label for the nvtx range.
    * @see NVTX_RANGE to use with minimal performance impact
    */
    explicit NVTXRange(const char *label) {
         util::nvtx::push(label);
    }
    /**
     *  Destructor which pops a marker off the nvtx stack.
     */
    ~NVTXRange() {
        util::nvtx::pop();
    }
};
};  // namespace nvtx
};  // namespace util

// If USE_NVTX is enabled, provide macros which actually use NVTX
#if defined(USE_NVTX)
/**
 * Macro which creates a scope-based NVTX range, with auto-popping of the marker.
 * If NVTX is defined, this constructs an util::nvtx::NVTXRange object with the specified label.
 * @param label the label for the NVTX marker.
 * @see util::nvtx::NVTXRange for implementation details
 */
#define NVTX_RANGE(label) util::nvtx::NVTXRange uniq_name_using_macros(label)
/**
 * Macro which pushes an NVTX marker onto the stack, if NVTX is defined.
 * @param label label for the NVTX marker
 * @see util::nvtx::push for implementation details.
 */
#define NVTX_PUSH(label) util::nvtx::push(label)
/**
 * Macro which pops an NVTX marker onto the stack, if NVTX is defined.
 * @see util::nvtx::pop for implementation details.
 */
#define NVTX_POP() util::nvtx::pop()
#else
// If NVTX is not enabled, provide macros which do nothing and optimise out any arguments.
// Documentation is for the enabled version for doxygen.
/**
 * Macro which creates a scope-based NVTX range, with auto-popping of the marker.
 * If NVTX is defined, this constructs an util::nvtx::NVTXRange object with the specified label.
 * @param label the label for the NVTX marker.
 * @see util::nvtx::NVTXRange for implementation details
 */
#define NVTX_RANGE(label)
/**
 * Macro which pushes an NVTX marker onto the stack, if NVTX is defined.
 * @param label label for the NVTX marker
 * @see util::nvtx::push for implementation details.
 */
#define NVTX_PUSH(label)
/**
 * Macro which pops an NVTX marker onto the stack, if NVTX is defined.
 * @see util::nvtx::pop for implementation details.
 */
#define NVTX_POP()
#endif

#endif  // INCLUDE_FLAMEGPU_UTIL_NVTX_H_
