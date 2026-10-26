#ifndef INCLUDE_FLAMEGPU_DETAIL_GPUEVENTTIMER_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_GPUEVENTTIMER_CUH_

#include "flamegpu/detail/Timer.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"
#include "flamegpu/detail/gpu/cuda_hip_dispatch.hpp"

namespace flamegpu {
namespace detail {

/**
 * Class to simplify the use of CUDA/HIP Events for timing.
 * Timing between CUDAEvent_t is only accurate in the default stream, hence streams cannot be passed.
 * @todo - investigate timing accurace on HIP (incl. on WDDM)
 * @todo - make this device aware (GetDevice, SetDevice)?
 * @todo - make this context aware, in case of DeviceReset between start and stop, or stop and getElapsed*
 */
class GPUEventTimer : public virtual Timer {
 public:
    /** 
     * Default constructor, creates the cuda/HIP Events and initialises values.
     */
    GPUEventTimer() :
    startEvent(NULL),
    stopEvent(NULL),
    ms(0.),
    startEventRecorded(false),
    stopEventRecorded(false),
    synced(false) {
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventCreate(&this->startEvent));
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventCreate(&this->stopEvent));
    }
    /** 
     * Destroys the cuda/HIP Events created by this instance
     * 
     * Todo: Make sure the context is still valid before destroying events
     */
    ~GPUEventTimer() {
        if (this->startEvent) {
            flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventDestroy(this->startEvent));
        }
        if (this->stopEvent) {
            flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventDestroy(this->stopEvent));
        }
        this->startEvent = NULL;
        this->stopEvent = NULL;
    }
    /**
     * Deleted copy ctors to prevent the same CUDA/HIP Event objects being owned by multiple timer objects.
     */
    GPUEventTimer(const GPUEventTimer&) = delete;
    GPUEventTimer& operator=(const GPUEventTimer&) = delete;
    /**
     * Move ctor which prevents events from being double-destroyed
     */
    GPUEventTimer(GPUEventTimer&& other) noexcept
        : startEvent(other.startEvent)
        , stopEvent(other.stopEvent) {
        other.startEvent = NULL;
        other.stopEvent  = NULL;
    }
    /**
     * Record the start event, resetting the syncronisation flag.
     */
    void start() override {
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventRecord(this->startEvent));
        this->startEventRecorded = true;
        this->stopEventRecorded = false;
        this->synced = false;
    }
    /**
     * Record the stop event, resetting the syncronisation flag.
     */
    void stop() override {
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventRecord(this->stopEvent));
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
     * Syncrhonize the cuda/HIP Event(s), calcualting the elapsed time in ms between the two events.
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
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventSynchronize(this->stopEvent));
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuEventElapsedTime(&this->ms, this->startEvent, this->stopEvent));
        synced = true;
    }

    /**
     * CUDA Event for the start event
     */
    flamegpu::detail::gpu::Event_t startEvent;
    /**
     * CUDA Event for the stop event
     */
    flamegpu::detail::gpu::Event_t stopEvent;
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

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPUEVENTTIMER_CUH_
