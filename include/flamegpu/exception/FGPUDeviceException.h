#ifndef INCLUDE_FLAMEGPU_EXCEPTION_FGPUDEVICEEXCEPTION_H_
#define INCLUDE_FLAMEGPU_EXCEPTION_FGPUDEVICEEXCEPTION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <type_traits>

#include "flamegpu/gpu/CUDAScanCompaction.h"

#ifndef NO_SEATBELTS

#include "flamegpu/exception/FGPUDeviceException_device.h"

/**
 * Host 'singleton', owned 1 per CUDAAgentModel, provides facility for generating and checking DeviceExceptionBuffers
 */
class DeviceExceptionManager {
 public:
    DeviceExceptionManager();
    /**
     * Free all device memory
     */
    ~DeviceExceptionManager();
    DeviceExceptionBuffer *getDevicePtr(const unsigned int &streamId);
    void checkError(const std::string &function, const unsigned int &streamId);

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
    DeviceExceptionBuffer *d_buffer[CUDAScanCompaction::MAX_STREAMS];
    /**
     * Host buffers to copy error buffers back to
     * 1 per stream
     */
    DeviceExceptionBuffer hd_buffer[CUDAScanCompaction::MAX_STREAMS];
};
#else
/**
 * Ignore the device error macro when NO_SEATBELTS is enabled
 * These checks are costly to performance
 */
#define DTHROW(nop)
#endif  // NO_SEATBELTS
#endif  // INCLUDE_FLAMEGPU_EXCEPTION_FGPUDEVICEEXCEPTION_H_
