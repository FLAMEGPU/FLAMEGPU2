#ifndef INCLUDE_FLAMEGPU_DETAIL_SIGNALHANDLERS_H_
#define INCLUDE_FLAMEGPU_DETAIL_SIGNALHANDLERS_H_
#include <cstdlib>
#include <csignal>

namespace flamegpu {
namespace detail {
/**
 * Signal handlers used to try and produce a clean exit on interrupt
 *
 * These currently don't do anything significantly different from default behaviour
 */
class SignalHandlers {
 private:
/**
 * Static method to handle SIGINT
 * Enables ctrl+c to close out of all active threads
 * @param signum The number of the signal
 */
static void handleSIGINT(int signum) {
    // Potentially do some graceful cleanup here. Not needed for now.
    // Close the application with the appropriate signal.
    std::exit(signum);
}
 public:
/**
 * Static method to register all included signal handlers
 */
static void registerSignalHandlers(){
    // Register the handler method for SIGINT
    std::signal(SIGINT, handleSIGINT);
}
};

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_SIGNALHANDLERS_H_
