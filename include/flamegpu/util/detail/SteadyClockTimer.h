#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_STEADYCLOCKTIMER_H_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_STEADYCLOCKTIMER_H_

#include <chrono>

namespace flamegpu {
namespace util {
namespace detail {

/** 
 * Class to simplify the finding of elapsed time using a chrono::steady_clock timer.
 */
class SteadyClockTimer {
 public:
    SteadyClockTimer() :
        _start(),
        _stop() { }

    ~SteadyClockTimer() { }

    void start() {
        _start = std::chrono::steady_clock::now();
    }

    void stop() {
        _stop = std::chrono::steady_clock::now();
    }

    float getElapsedMilliseconds() {
        std::chrono::duration<double> elapsed = this->_stop - this->_start;
        float ms = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        return ms;
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> _start;
    std::chrono::time_point<std::chrono::steady_clock> _stop;
};

}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_STEADYCLOCKTIMER_H_
