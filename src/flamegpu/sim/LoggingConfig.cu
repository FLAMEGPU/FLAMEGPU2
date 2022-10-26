#include "flamegpu/sim/LoggingConfig.h"

#include "flamegpu/sim/AgentLoggingConfig.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"

namespace flamegpu {

LoggingConfig::LoggingConfig(const ModelDescription &_model)
    :model(_model.model->clone())
    , log_timing(false) { }
LoggingConfig::LoggingConfig(const ModelData &_model)
    :model(_model.clone())
    , log_timing(false) { }
LoggingConfig::LoggingConfig(const LoggingConfig &other)
    : model(other.model->clone())
    , environment(other.environment)
    , agents(other.agents)
    , log_timing(other.log_timing) { }
AgentLoggingConfig LoggingConfig::agent(const std::string &agent_name, const std::string &agent_state) {
    // Validate the agent state combination exists
    auto model_agent_it = model->agents.find(agent_name);
    if (model_agent_it == model->agents.end()) {
        THROW exception::InvalidAgentName("Agent '%s' was not found in the model description, "
            "in LoggingConfig::agent()\n",
            agent_name.c_str());
    }
    if (model_agent_it->second->states.find(agent_state) == model_agent_it->second->states.end()) {
        THROW exception::InvalidAgentState("State '%s' was not found within agent '%s' in the model description, "
            "in LoggingConfig::agent()\n",
            agent_state.c_str(), agent_name.c_str());
    }
    util::StringPair name = std::make_pair(agent_name, agent_state);
    auto agent_it = agents.find(name);
    if (agent_it== agents.end())
        agent_it = agents.emplace(name, std::make_pair(std::make_shared<std::set<NameReductionFn>>(), false)).first;
    return AgentLoggingConfig(model_agent_it->second, agent_it->second);
}

void LoggingConfig::logEnvironment(const std::string &property_name) {
    // Validate the environment property exists
    auto env_map = model->environment->getPropertiesMap();
    if (env_map.find(property_name) == env_map.end()) {
        THROW exception::InvalidEnvProperty("Environment property '%s' was not found in the model description, "
            "in LoggingConfig::logEnvironment()\n",
            property_name.c_str());
    }
    // Log the property
    if (!environment.emplace(property_name).second) {
        THROW exception::InvalidEnvProperty("Environment property '%s' has already been marked for logging, "
            "in LoggingConfig::logEnvironment()\n",
            property_name.c_str());
    }
}
void LoggingConfig::logTiming(bool doLogTiming) {
    log_timing = doLogTiming;
}
StepLoggingConfig::StepLoggingConfig(const ModelDescription &model)
    : LoggingConfig(model)
    , frequency(1) { }
StepLoggingConfig::StepLoggingConfig(const ModelData &model)
    : LoggingConfig(model)
    , frequency(1) { }
StepLoggingConfig::StepLoggingConfig(const StepLoggingConfig &other)
    : LoggingConfig(other)
    , frequency(other.frequency) { }
StepLoggingConfig::StepLoggingConfig(const LoggingConfig &other)
    : LoggingConfig(other)
    , frequency(1) { }
void StepLoggingConfig::setFrequency(const unsigned int steps) {
    frequency = steps;
}

}  // namespace flamegpu
