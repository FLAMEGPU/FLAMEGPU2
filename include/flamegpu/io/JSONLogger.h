#ifndef INCLUDE_FLAMEGPU_IO_JSONLOGGER_H_
#define INCLUDE_FLAMEGPU_IO_JSONLOGGER_H_

#include <string>
#include <typeindex>

#include <nlohmann/json.hpp>

#include "flamegpu/io/Logger.h"
#include "flamegpu/detail/Any.h"

namespace flamegpu {
struct RunLog;
struct StepLogFrame;
struct ExitLogFrame;
struct LogFrame;
class RunPlan;

namespace io {

/**
 * JSON format Logger
 */
class JSONLogger : public Logger{
 public:
    JSONLogger(const std::string &outPath, bool prettyPrint, bool truncateFile);
    /**
     * Log a runlog to file, using a RunPlan in place of config
     * @throws May throw exceptions if logging to file failed for any reason
     */
    void log(const RunLog &log, const RunPlan &plan, bool logSteps = true, bool logExit = true, bool logStepTime = false, bool logExitTime = false) const override;
    /**
     * Log a runlog to file, uses config data (random seed) from the RunLog
     * @throws May throw exceptions if logging to file failed for any reason
     */
    void log(const RunLog &log, bool logConfig = true, bool logSteps = true, bool logExit = true, bool logStepTime = false, bool logExitTime = false) const override;

 private:
    /**
     * Internal logging method, allows Plan to be passed as null
     */
    void logCommon(const RunLog &log, const RunPlan *plan, bool logConfig, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const;

    void logCommon(nlohmann::ordered_json& j, const RunLog &log, const RunPlan *plan, bool logConfig, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const;
    /**
     * Returns the run config as a JSON object
     * @param log RunLog containing the config items to be written
     */
    nlohmann::ordered_json logConfig(const RunLog &log) const;
    /**
     * Returns the run plan as a JSON object
     * @param plan RunPlan containing the config items to be written
     */
    nlohmann::ordered_json logConfig(const RunPlan &plan) const;
    /**
     * Return a json object containing performance specs
     * @param log RunLog containing the config items to be written
     */
    nlohmann::ordered_json logPerformanceSpecs(const RunLog& log) const;
    /**
     * Writes out step logs as a JSON array via the provided writer
     * @param j nhlomann::json instance
     * @param log RunLog containing the step logs to be written
     * @param logTime Include time in the log to be written
     */
    void logSteps(nlohmann::ordered_json& j, const RunLog &log, bool logTime) const;
    /**
     * Writes out an exit log as a JSON object via the provided writer
     * @param j nhlomann::json instance
     * @param log RunLog containing the exit log to be written
     * @param logTime Include time in the log to be written
     */
    void logExit(nlohmann::ordered_json& j, const RunLog &log, bool logTime) const;
    /**
     * Returns the StepLogFrame as a JSON object
     * @param log LogFrame to be written
     * @param logTime Include time in the log to be written
     */
    nlohmann::ordered_json writeLogFrame(const StepLogFrame&log, bool logTime) const;
    /**
     * Returns the ExitLogFrame as a JSON object
     * @param log LogFrame to be written
     * @param logTime Include time in the log to be written
     */
    nlohmann::ordered_json writeLogFrame(const ExitLogFrame& log, bool logTime) const;
    /**
     * Writes out a LogFrame instance as a JSON object via the provided writer
     * @param j nhlomann::json instance
     * @param log LogFrame to be written
     */
    void writeCommonLogFrame(nlohmann::ordered_json& j, const LogFrame& log) const;
    /**
     * Writes out the value of an Any via the provided writer
     * @param j nhlomann::json instance
     * @param value The Any to be written
     * @param elements The number of individual elements stored in the Any (1 if not an array)
     */
    void writeAny(nlohmann::ordered_json& j, const detail::Any &value, unsigned int elements = 1) const;

    std::string out_path;
    bool prettyPrint;
    bool truncateFile;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONLOGGER_H_
