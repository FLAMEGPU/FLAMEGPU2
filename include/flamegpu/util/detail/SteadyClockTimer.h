#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_STEADYCLOCKTIMER_H_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_STEADYCLOCKTIMER_H_

#include <chrono>

#include "flamegpu/util/detail/Timer.h"
#include "flamegpu/exception/FLAMEGPUException.h"


namespace flamegpu {
namespace util {
namespace detail {

/** 
 * Class to simplify the finding of elapsed time using a chrono::steady_clock timer.
 */
class SteadyClockTimer : public virtual Timer {
 public:
    /** 
     * Default constructor, initialising values.
     */
    SteadyClockTimer() :
        startTime(),
        stopTime(),
        startEventRecorded(false),
        stopEventRecorded(false) { }

    /** 
     * Default destructor
     */
    ~SteadyClockTimer() { }

    /**
     * Record the start steady clock time
     */ 
    void start() {
        this->startTime = std::chrono::steady_clock::now();
        this->startEventRecorded = true;
        this->stopEventRecorded = false;
    }

    /**
     * Record the start steady clock stop time
     */
    void stop() {
        this->stopTime = std::chrono::steady_clock::now();
        this->stopEventRecorded = true;
    }

    /**
     * Get the elapsed time between calls to start() and stop() in milliseconds
     * @return elapsed time in milliseconds
     */
    float getElapsedMilliseconds() {
        if (!startEventRecorded) {
            THROW exception::TimerException("start() must be called prior to getElapsed*");
        }
        if (!stopEventRecorded) {
            THROW exception::TimerException("stop() must be called prior to getElapsed*");
        }
        std::chrono::duration<double> elapsed = this->stopTime - this->startTime;
        float ms = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        return ms;
    }

    /**
     * Get the elapsed time between calls to start() and stop() in seconds
     * @return elapsed time in milliseconds
     */
    float getElapsedSeconds() {
        return this->getElapsedMilliseconds() / 1000.0f;
    }

 private:
    /** 
     * Time point for the start event
     */
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    /** 
     * Time point for the stop event
     */
    std::chrono::time_point<std::chrono::steady_clock> stopTime;
    /**
     * Flag indicating if the start event has been recorded or not.
     */
    bool startEventRecorded;
    /**
     * Flag indicating if the start event has been recorded or not.
     */
    bool stopEventRecorded;
};

}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_STEADYCLOCKTIMER_H_
