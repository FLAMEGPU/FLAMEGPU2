#include "flamegpu/io/JSONLogger.h"

#include <iostream>
#include <fstream>
#include <string>

#include "flamegpu/simulation/RunPlan.h"
#include "flamegpu/simulation/LogFrame.h"

namespace flamegpu {
namespace io {

JSONLogger::JSONLogger(const std::string &outPath, bool _prettyPrint, bool _truncateFile)
    : out_path(outPath)
    , prettyPrint(_prettyPrint)
    , truncateFile(_truncateFile) { }

void JSONLogger::log(const RunLog &log, const RunPlan &plan, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const {
  logCommon(log, &plan, false, logSteps, logExit, logStepTime, logExitTime);
}
void JSONLogger::log(const RunLog &log, bool logConfig, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const {
  logCommon(log, nullptr, logConfig, logSteps, logExit, logStepTime, logExitTime);
}

void JSONLogger::writeAny(nlohmann::ordered_json& j, const detail::Any &value, const unsigned int elements) const {
    if (elements == 1) {
        if (value.type == std::type_index(typeid(float))) {
            j = static_cast<const float*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(double))) {
            j = static_cast<const double*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(int64_t))) {
            j = static_cast<const int64_t*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(uint64_t))) {
            j = static_cast<const uint64_t*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(int32_t))) {
            j = static_cast<const int32_t*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(uint32_t))) {
            j = static_cast<const uint32_t*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(int16_t))) {
            j = static_cast<const int16_t*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(uint16_t))) {
            j = static_cast<const uint16_t*>(value.ptr)[0];
        } else if (value.type == std::type_index(typeid(int8_t))) {
            j = static_cast<int32_t>(static_cast<const int8_t*>(value.ptr)[0]);  // Char outputs weird if being used as an integer
        } else if (value.type == std::type_index(typeid(uint8_t))) {
            j = static_cast<uint32_t>(static_cast<const uint8_t*>(value.ptr)[0]);  // Char outputs weird if being used as an integer
        } else if (value.type == std::type_index(typeid(char))) {
            j = static_cast<int32_t>(static_cast<const char*>(value.ptr)[0]);  // Char outputs weird if being used as an integer
        } else {
            THROW exception::JSONError("Attempting to export value of unsupported type '%s', "
                "in JSONLogger::writeAny()\n", value.type.name());
        }
        return;
    }
    // Loop through elements, to construct array
    for (unsigned int el = 0; el < elements; ++el) {
        if (value.type == std::type_index(typeid(float))) {
            j.emplace_back(static_cast<const float*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(double))) {
            j.emplace_back(static_cast<const double*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(int64_t))) {
            j.emplace_back(static_cast<const int64_t*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(uint64_t))) {
            j.emplace_back(static_cast<const uint64_t*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(int32_t))) {
            j.emplace_back(static_cast<const int32_t*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(uint32_t))) {
            j.emplace_back(static_cast<const uint32_t*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(int16_t))) {
            j.emplace_back(static_cast<const int16_t*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(uint16_t))) {
            j.emplace_back(static_cast<const uint16_t*>(value.ptr)[el]);
        } else if (value.type == std::type_index(typeid(int8_t))) {
            j.emplace_back(static_cast<int32_t>(static_cast<const int8_t*>(value.ptr)[el]));  // Char outputs weird if being used as an integer
        } else if (value.type == std::type_index(typeid(uint8_t))) {
            j.emplace_back(static_cast<uint32_t>(static_cast<const uint8_t*>(value.ptr)[el]));  // Char outputs weird if being used as an integer
        } else if (value.type == std::type_index(typeid(char))) {
            j.emplace_back(static_cast<int32_t>(static_cast<const char*>(value.ptr)[el]));  // Char outputs weird if being used as an integer
        } else {
            THROW exception::JSONError("Attempting to export value of unsupported type '%s', "
                "in JSONLogger::writeAny()\n", value.type.name());
        }
    }
}
nlohmann::ordered_json JSONLogger::writeLogFrame(const StepLogFrame& frame, bool logTime) const {
    nlohmann::ordered_json j;
    if (logTime) {
        j["step_time"] = frame.getStepTime();
    }
    writeCommonLogFrame(j, frame);
    return j;
}
nlohmann::ordered_json JSONLogger::writeLogFrame(const ExitLogFrame& frame, bool logTime) const {
    nlohmann::ordered_json j;
    if (logTime) {
        j["rtc_time"] = frame.getRTCTime();
        j["init_time"] = frame.getInitTime();
        j["exit_time"] = frame.getExitTime();
        j["total_time"] = frame.getTotalTime();
    }
    writeCommonLogFrame(j, frame);
    return j;
}
void JSONLogger::writeCommonLogFrame(nlohmann::ordered_json& j, const LogFrame &frame) const {
    // Add static items
    j["step_index"] = frame.getStepCount();
    if (frame.getEnvironment().size()) {
        // Add dynamic environment values
        nlohmann::ordered_json j_env;
        for (const auto &prop : frame.getEnvironment()) {
            j_env[prop.first] = {};
            // Log value
            writeAny(j_env[prop.first], prop.second, prop.second.elements);
        }
        j["env"] = j_env;
    }

    if (frame.getAgents().size()) {
        // Add dynamic agent values
        nlohmann::ordered_json j_agents = {};
        for (const auto &agent : frame.getAgents()) {
            nlohmann::ordered_json j_t_agent;
            // Log agent count if provided
            if (agent.second.second != UINT_MAX) {
                j_t_agent["count"][agent.second.second];
            }
            if (agent.second.first.size()) {
                j_t_agent["variables"];
                // This assumes that sort order places all variables of same name, different reduction consecutively
                std::string current_variable;
                // Log each reduction
                for (auto &var : agent.second.first) {
                    // Log value
                    writeAny(j_t_agent["variables"][var.first.name][LoggingConfig::toString(var.first.reduction)], var.second, 1);
                }
            }
            j_agents[agent.first.first][agent.first.second].push_back(j_t_agent);
        }
        j["agents"] = j_agents;
    }
}

nlohmann::ordered_json JSONLogger::logConfig(const RunLog &log) const {
    nlohmann::ordered_json j;
    j["random_seed"] = log.getRandomSeed();
    return j;
}
nlohmann::ordered_json JSONLogger::logConfig(const RunPlan &plan) const {
    nlohmann::ordered_json j;
    // Add static items
    j["random_seed"] = plan.getRandomSimulationSeed();
    j["steps"] = plan.getSteps();
    // Add dynamic environment overrides
    nlohmann::ordered_json dyn_j;
    for (const auto& prop : plan.property_overrides) {
        const EnvironmentData::PropData& env_prop = plan.environment->at(prop.first);
        dyn_j[prop.first] = {};
        writeAny(dyn_j[prop.first], prop.second, env_prop.data.elements);
    }
    j["environment"] = dyn_j;
    return j;
}
nlohmann::ordered_json JSONLogger::logPerformanceSpecs(const RunLog& log) const {
    nlohmann::ordered_json j;
    // Add static items
    j["device_name"] = log.getPerformanceSpecs().device_name;
    j["device_cc_major"] = log.getPerformanceSpecs().device_cc_major;
    j["device_cc_minor"] = log.getPerformanceSpecs().device_cc_minor;
    j["cuda_version"] = log.getPerformanceSpecs().cuda_version;
    j["seatbelts"] = log.getPerformanceSpecs().seatbelts;
    j["flamegpu_version"] = log.getPerformanceSpecs().flamegpu_version;
    return j;
}
void JSONLogger::logSteps(nlohmann::ordered_json& j, const RunLog &log, bool logTime) const {
    j["steps"] = {};
    for (const auto &step : log.getStepLog()) {
        j["steps"].push_back(writeLogFrame(step, logTime));
    }
}
void JSONLogger::logExit(nlohmann::ordered_json& j, const RunLog &log, bool logTime) const {
    j["exit"] = writeLogFrame(log.getExitLog(), logTime);
}

void JSONLogger::logCommon(nlohmann::ordered_json &j, const RunLog &log, const RunPlan *plan, bool doLogConfig, bool doLogSteps, bool doLogExit, bool doLogStepTime, bool doLogExitTime) const {
    // Log config
    if (plan) {
        j["config"] = logConfig(*plan);
    } else if (doLogConfig) {
        j["config"] = logConfig(log);
    }
    if (doLogStepTime || doLogExitTime) {
        j["performance_specs"] = logPerformanceSpecs(log);
    }

    // Log step log
    if (doLogSteps) {
        logSteps(j, log, doLogStepTime);
    }

    // Log exit log
    if (doLogExit) {
        logExit(j, log, doLogExitTime);
    }
}
void JSONLogger::logCommon(const RunLog &log, const RunPlan *plan, bool doLogConfig, bool doLogSteps, bool doLogExit, bool doLogStepTime, bool doLogExitTime) const {
    // Init writer
    nlohmann::ordered_json j;
    logCommon(j, log, plan, doLogConfig, doLogSteps, doLogExit, doLogStepTime, doLogExitTime);
    // Perform output
    std::ofstream out(out_path, std::ios::binary | (truncateFile ? std::ofstream::trunc : std::ofstream::app));
    if (!out.is_open()) {
        THROW exception::JSONError("Unable to open file '%s' for writing\n", out_path.c_str());
    }
    if (prettyPrint) {
        out << std::setw(4);
    }
    out << j;
    out.close();
}

}  // namespace io
}  // namespace flamegpu
