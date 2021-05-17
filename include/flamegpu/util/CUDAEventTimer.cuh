#ifndef INCLUDE_FLAMEGPU_UTIL_CUDAEVENTTIMER_CUH_
#define INCLUDE_FLAMEGPU_UTIL_CUDAEVENTTIMER_CUH_

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

namespace flamegpu {
namespace util {

/**
 * Class to simplify the use of CUDAEvents for timing.
 * Timing between CUDAEvent_t is only accurate in the default stream, hence streams cannot be passed.
 * @todo - this appears unreliable on WDDM devices
 * @todo - make this device aware (cudaGetDevice, cudaSetDevice)?
 */
class CUDAEventTimer {
 public:
    /** 
     * Default constructor, creates the cudaEvents and initialises values.
     */
    CUDAEventTimer() :
    startEvent(NULL),
    stopEvent(NULL),
    ms(0.),
    synced(false) {
        gpuErrchk(cudaEventCreate(&this->startEvent));
        gpuErrchk(cudaEventCreate(&this->stopEvent));
    }
    /** 
     * Destroys the cudaEvents created by this instance
     */
    ~CUDAEventTimer() {
        gpuErrchk(cudaEventDestroy(this->startEvent));
        gpuErrchk(cudaEventDestroy(this->stopEvent));
        this->startEvent = NULL;
        this->startEvent = NULL;
    }
    /**
     * Record the start event, resetting the syncronisation flag.
     */
    void start() {
        gpuErrchk(cudaEventRecord(this->startEvent));
        synced = false;
    }
    /**
     * Record the stop event, resetting the syncronisation flag.
     */
    void stop() {
        gpuErrchk(cudaEventRecord(this->stopEvent));
        synced = false;
    }
    /**
     * Syncrhonize the cudaEvent(s), calcualting the elapsed time in ms between the two events.
     * this is only accurate if used for the default stream (hence streams are not used)
     * Sets the flag indicating syncronisation has occured, and therefore the elapsed time can be queried.
     * @return elapsed time in milliseconds
     */
    float sync() {
        gpuErrchk(cudaEventSynchronize(this->stopEvent));
        gpuErrchk(cudaEventElapsedTime(&this->ms, this->startEvent, this->stopEvent));
        synced = true;
        return ms;
    }
    /**
     * Get the elapsed time between the start event being issued and the stop event occuring.
     * @return elapsed time in milliseconds
     */
    float getElapsedMilliseconds() {
        if (!synced) {
            THROW UnsycnedCUDAEventTimer();
        }
        return ms;
    }

 private:
    /**
     * CUDA Event for the start event
     */
    cudaEvent_t startEvent;
    /**
     * CUDA Event for the stop event
     */
    cudaEvent_t stopEvent;
    /**
     * Elapsed times between start and stop in milliseconds
     */
    float ms;
    /**
     * Flag to return whether events have been synced or not.
     */
    bool synced;
};

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_CUDAEVENTTIMER_CUH_
