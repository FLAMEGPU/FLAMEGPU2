#ifndef INCLUDE_FLAMEGPU_DETAIL_CUDAEVENTTIMER_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_CUDAEVENTTIMER_CUH_

#include <cuda_runtime.h>

#include "flamegpu/detail/Timer.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

namespace flamegpu {
namespace detail {

/**
 * Class to simplify the use of CUDAEvents for timing.
 * Timing between CUDAEvent_t is only accurate in the default stream, hence streams cannot be passed.
 * @todo - this appears unreliable on WDDM devices
 * @todo - make this device aware (cudaGetDevice, cudaSetDevice)?
 * @todo - make this context aware, in case of cudaDeviceReset between start and stop, or stop and getElapsed*
 */
class CUDAEventTimer : public virtual Timer {
 public:
    /** 
     * Default constructor, creates the cudaEvents and initialises values.
     */
    CUDAEventTimer() :
    startEvent(NULL),
    stopEvent(NULL),
    ms(0.),
    startEventRecorded(false),
    stopEventRecorded(false),
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
        this->stopEvent = NULL;
    }
    /**
     * Record the start event, resetting the syncronisation flag.
     */
    void start() override {
        gpuErrchk(cudaEventRecord(this->startEvent));
        this->startEventRecorded = true;
        this->stopEventRecorded = false;
        this->synced = false;
    }
    /**
     * Record the stop event, resetting the syncronisation flag.
     */
    void stop() override {
        gpuErrchk(cudaEventRecord(this->stopEvent));
        this->stopEventRecorded = true;
        this->synced = false;
    }
    /**
     * Get the elapsed time between the start event being issued and the stop event occuring.
     * @return elapsed time in milliseconds
     */
    float getElapsedMilliseconds() override {
        // If the cuda event timer has not been synchonised, sync it. This stores the time internally.
        if (!this->synced) {
            this->sync();
        }
        // Return the stored elapsed time in milliseconds.
        return this->ms;
    }

    /**
     * Get the elapsed time between the start event being issued and the stop event occuring.
     * @return elapsed time in seconds
     */
    float getElapsedSeconds() override {
        // Get the elapsed time in milliseconds, and convert it to seconds.
        return this->getElapsedMilliseconds() / 1000.0f;
    }



 private:
    /**
     * Syncrhonize the cudaEvent(s), calcualting the elapsed time in ms between the two events.
     * this is only accurate if used for the default stream (hence streams are not used)
     * Sets the flag indicating syncronisation has occured, and therefore the elapsed time can be queried.
     * @return elapsed time in milliseconds
     */
    void sync() {
        // If the start or stop events have not yet been recorded, do not proceed and throw an exception.
        if (!startEventRecorded) {
            THROW exception::TimerException("start() must be called prior to getElapsed*");
        }
        if (!stopEventRecorded) {
            THROW exception::TimerException("stop() must be called prior to getElapsed*");
        }
        gpuErrchk(cudaEventSynchronize(this->stopEvent));
        gpuErrchk(cudaEventElapsedTime(&this->ms, this->startEvent, this->stopEvent));
        synced = true;
    }

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
     * Flag indicating if the start event has been recorded or not.
     */
    bool startEventRecorded;
    /**
     * Flag indicating if the start event has been recorded or not.
     */
    bool stopEventRecorded;
    /**
     * Flag to return whether events have been synced or not.
     */
    bool synced;
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_CUDAEVENTTIMER_CUH_
