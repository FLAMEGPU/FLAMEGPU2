#ifndef INCLUDE_FLAMEGPU_IO_LOGGER_H_
#define INCLUDE_FLAMEGPU_IO_LOGGER_H_

struct RunLog;
class RunPlan;

class Logger {
 public:
    virtual ~Logger() = default;
    /**
     * Log a runlog to file, using a RunPlan in place of config
     * @throws May throw exceptions if logging to file failed for any reason
     */
    virtual void log(const RunLog &log, const RunPlan &plan, bool logSteps = true, bool logExit = true) const = 0;
    /**
     * Log a runlog to file, uses config data (random seed) from the RunLog
     * @throws May throw exceptions if logging to file failed for any reason
     */
    virtual void log(const RunLog &log, bool logConfig = true, bool logSteps = true, bool logExit = true) const = 0;
};

#endif  // INCLUDE_FLAMEGPU_IO_LOGGER_H_
