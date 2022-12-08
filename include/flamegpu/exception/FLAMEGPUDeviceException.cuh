#ifndef INCLUDE_FLAMEGPU_EXCEPTION_FLAMEGPUDEVICEEXCEPTION_CUH_
#define INCLUDE_FLAMEGPU_EXCEPTION_FLAMEGPUDEVICEEXCEPTION_CUH_

#include <string>
#include <type_traits>

#include "flamegpu/simulation/detail/CUDAScanCompaction.h"

#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS

#include "flamegpu/exception/FLAMEGPUDeviceException_device.cuh"

namespace flamegpu {
namespace exception {

/**
 * Host 'singleton', owned 1 per CUDASimulation, provides facility for generating and checking DeviceExceptionBuffers
 */
class DeviceExceptionManager {
 public:
    DeviceExceptionManager();
    /**
     * Free all device memory
     */
    ~DeviceExceptionManager();
    DeviceExceptionBuffer *getDevicePtr(unsigned int streamId, cudaStream_t stream);
    void checkError(const std::string &function, unsigned int streamId, cudaStream_t stream);

 private:
    /**
     * Generate a string representing the throw location
     */
    static std::string getLocationString(const DeviceExceptionBuffer &b);
    /**
     * Generate a string representing the error message
     */
    static std::string getErrorString(const DeviceExceptionBuffer &b);
    /**
     * Pointers to device buffers for error reporting
     * 1 per stream
     * nullptr until used
     */
    DeviceExceptionBuffer *d_buffer[detail::CUDAScanCompaction::MAX_STREAMS];
    /**
     * Host buffers to copy error buffers back to
     * 1 per stream
     */
    DeviceExceptionBuffer hd_buffer[detail::CUDAScanCompaction::MAX_STREAMS];
};
}  // namespace exception
}  // namespace flamegpu
#else
/**
 * Ignore the device error macro when FLAMEGPU_SEATBELTS are off
 * These checks are costly to performance
 */
#define DTHROW(nop)
#endif  // FLAMEGPU_SEATBELTS=OFF
#endif  // INCLUDE_FLAMEGPU_EXCEPTION_FLAMEGPUDEVICEEXCEPTION_CUH_
