#ifndef INCLUDE_FLAMEGPU_IO_JSONLOGGER_H_
#define INCLUDE_FLAMEGPU_IO_JSONLOGGER_H_

#include <string>
#include <typeindex>

#include "flamegpu/io/Logger.h"
#include "flamegpu/util/Any.h"

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
    /**
     * rapidjson::Writer doesn't have virtual methods, so can't pass rapidjson::PrettyWriter around as ptr to rapidjson::writer
     * Instead we call a templated version of all the methods
     */
    template<typename T>
    void logCommon(T &writer, const RunLog &log, const RunPlan *plan, bool logConfig, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const;
    /**
     * Writes out the run config via a JSON object
     * @param writer Rapidjson writer instance
     * @param log RunLog containing the config items to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void logConfig(T &writer, const RunLog &log) const;
    /**
     * Writes out step logs as a JSON array via the provided writer
     * @param writer Rapidjson writer instance
     * @param plan RunPlan containing the config items to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void logConfig(T &writer, const RunPlan &plan) const;
    /**
     * Writes out step logs as a JSON array via the provided writer
     * @param writer Rapidjson writer instance
     * @param log RunLog containing the config items to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void logPerformanceSpecs(T& writer, const RunLog& log) const;
    /**
     * Writes out step logs as a JSON array via the provided writer
     * @param writer Rapidjson writer instance
     * @param log RunLog containing the step logs to be written
     * @param logTime Include time in the log to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void logSteps(T &writer, const RunLog &log, bool logTime) const;
    /**
     * Writes out an exit log as a JSON object via the provided writer
     * @param writer Rapidjson writer instance
     * @param log RunLog containing the exit log to be written
     * @param logTime Include time in the log to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void logExit(T &writer, const RunLog &log, bool logTime) const;
    /**
     * Writes out an StepLogFrame instance as a JSON object via the provided writer
     * @param writer Rapidjson writer instance
     * @param log LogFrame to be written
     * @param logTime Include time in the log to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void writeLogFrame(T &writer, const StepLogFrame&log, bool logTime) const;
    /**
     * Writes out an ExitLogFrame instance as a JSON object via the provided writer
     * @param writer Rapidjson writer instance
     * @param log LogFrame to be written
     * @param logTime Include time in the log to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void writeLogFrame(T& writer, const ExitLogFrame& log, bool logTime) const;
    /**
     * Writes out a LogFrame instance as a JSON object via the provided writer
     * @param writer Rapidjson writer instance
     * @param log LogFrame to be written
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void writeCommonLogFrame(T& writer, const LogFrame& log) const;
    /**
     * Writes out the value of an Any via the provided writer
     * @param writer Rapidjson writer instance
     * @param value The Any to be written
     * @param elements The number of individual elements stored in the Any (1 if not an array)
     * @tparam T Instance of rapidjson::Writer or subclass (e.g. rapidjson::PrettyWriter)
     * @note Templated as can't forward declare rapidjson::Writer<rapidjson::StringBuffer>
     */
    template<typename T>
    void writeAny(T &writer, const util::Any &value, unsigned int elements = 1) const;

    std::string out_path;
    bool prettyPrint;
    bool truncateFile;
};
}  // namespace io
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_IO_JSONLOGGER_H_
