#include <utility>

#include "flamegpu/sim/AgentLoggingConfig.h"
#include "flamegpu/model/AgentData.h"

namespace flamegpu_internal {
    __constant__ double STANDARD_DEVIATION_MEAN;
    std::mutex STANDARD_DEVIATION_MEAN_mutex;
    standard_deviation_add_impl standard_deviation_add;
    standard_deviation_subtract_mean_impl standard_deviation_subtract_mean;
}

AgentLoggingConfig::AgentLoggingConfig(
    std::shared_ptr<const AgentData> _agent,
    std::pair<std::shared_ptr<std::set<LoggingConfig::NameReductionFn>>, bool> &_agent_set)
    : agent(std::move(_agent))
    , agent_set(_agent_set.first)
    , log_count(_agent_set.second) { }

void AgentLoggingConfig::log(const LoggingConfig::NameReductionFn &nrf, const std::type_index &variable_type, const std::string &method_name) {
    // Validate variable name and type
    const auto var = agent->variables.find(nrf.name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Agent ('%s') variable '%s' was not found in the model description, "
            "in AgentLoggingConfig::log%s()\n",
            agent->name.c_str(), nrf.name.c_str(), method_name.c_str());
    } else if (var->second.type != variable_type) {
        THROW InvalidVarType("Agent ('%s') variable '%s' has type '%s', incorrect type '%s' was provided to template, "
            "in AgentLoggingConfig::log%s()\n",
            agent->name.c_str(), nrf.name.c_str(), var->second.type.name(), variable_type.name(), method_name.c_str());
    } else if (var->second.elements != 1) {
        THROW InvalidVarType("Agent ('%s') variable '%s' is an array variable, this function does not support array variables, "
            "in AgentLoggingConfig::log%s()\n",
            agent->name.c_str(), nrf.name.c_str(), method_name.c_str());
    }
    // Store it in the map
    if (!agent_set->emplace(nrf).second) {
        THROW InvalidArgument("Agent ('%s') variable '%s' %s has already been marked for logging, "
            "in AgentLoggingConfig::log%s()\n",
             agent->name.c_str(), nrf.name.c_str(), method_name.c_str(), method_name.c_str());
    }
}
