#ifndef INCLUDE_FLAMEGPU_DETAIL_TIMER_H_
#define INCLUDE_FLAMEGPU_DETAIL_TIMER_H_


namespace flamegpu {
namespace detail {

/** 
 * Virtual Base Timer class used to standardise the API for different timer implementations.
 */
class Timer {
 public:
    /**
     * Default Destructor
     */
    virtual ~Timer() = default;

    /**
     * Start recording with this timer
     */
    virtual void start() = 0;

    /**
     * Stop recording the timer.
     */
    virtual void stop() = 0;

    /**
     * Get the time elapsed in milliseconds
     */
    virtual float getElapsedMilliseconds() = 0;

    /**
     * Get the time elapsed in seconds
     */
    virtual float getElapsedSeconds() = 0;
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_TIMER_H_
