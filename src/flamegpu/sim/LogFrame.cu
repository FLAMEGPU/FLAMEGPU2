#include "flamegpu/sim/LogFrame.h"

namespace flamegpu {

LogFrame::LogFrame()
    : step_count(0) { }


LogFrame::LogFrame(const std::map<std::string, util::Any> &_environment,
const std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>> &_agents,
const unsigned int &_step_count)
    : environment(_environment)
    , agents(_agents)
    , step_count(_step_count) { }


bool LogFrame::hasEnvironmentProperty(const std::string &property_name) const {
    const auto &it = environment.find(property_name);
    return it != environment.end();
}

AgentLogFrame LogFrame::getAgent(const std::string &agent_name, const std::string &state_name) const {
    const auto &it = agents.find({agent_name, state_name});
    if (it == agents.end()) {
          THROW exception::InvalidAgentState("Log data for agent '%s' state '%s' was not found, "
              "in LogFrame::getEnvironmentProperty()\n",
              agent_name.c_str(), state_name.c_str());
    }
    return AgentLogFrame(it->second.first, it->second.second);
}

AgentLogFrame::AgentLogFrame(const std::map<LoggingConfig::NameReductionFn, util::Any> &_data, const unsigned int &_count)
    : data(_data)
    , count(_count) { }


unsigned int AgentLogFrame::getCount() const {
    if (count != UINT_MAX)
        return count;
    THROW exception::InvalidOperation("Count of agents in state was not found in the log, "
        "in AgentLogFrame::getCount()\n");
}
double AgentLogFrame::getMean(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::Mean});
    if (it == data.end()) {
        THROW exception::InvalidAgentVar("Mean of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getMean()\n",
            variable_name.c_str());
    }
    return *static_cast<double *>(it->second.ptr);
}
double AgentLogFrame::getStandardDev(const std::string &variable_name) const {
    const auto &it = data.find({variable_name, LoggingConfig::StandardDev});
    if (it == data.end()) {
        THROW exception::InvalidAgentVar("Standard deviation of agent variable '%s' was not found in the log, "
            "in AgentLogFrame::getStandardDev()\n",
            variable_name.c_str());
    }
    return *static_cast<double *>(it->second.ptr);
}

StepLogFrame::StepLogFrame()
    : LogFrame()
    , step_time(0.0) { }


StepLogFrame::StepLogFrame(const std::map<std::string, util::Any>&& _environment,
    const std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>>&& _agents,
    const unsigned int& _step_count)
    : LogFrame(_environment, _agents, _step_count)
    , step_time(0.0) { }

ExitLogFrame::ExitLogFrame()
    : LogFrame()
    , rtc_time(0.0)
    , init_time(0.0)
    , exit_time(0.0)
    , total_time(0.0) { }


ExitLogFrame::ExitLogFrame(const std::map<std::string, util::Any>&& _environment,
    const std::map<util::StringPair, std::pair<std::map<LoggingConfig::NameReductionFn, util::Any>, unsigned int>>&& _agents,
    const unsigned int& _step_count)
    : LogFrame(_environment, _agents, _step_count)
    , rtc_time(0.0)
    , init_time(0.0)
    , exit_time(0.0)
    , total_time(0.0) { }

}  // namespace flamegpu
