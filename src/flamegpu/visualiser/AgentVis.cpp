#include "flamegpu/visualiser/AgentVis.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/model/AgentData.h"
#include "FLAMEGPU_Visualisation.h"  // TODO - This should probably be flamegpu_visualiser/FLAMEGPU_Visualisation.h?

AgentVis::AgentVis(CUDAAgent &_agent)
    : defaultConfig()
    , agent(_agent)
    , agentData(_agent.getAgentDescription())
    , x_var(_agent.getAgentDescription().variables.find("x") != _agent.getAgentDescription().variables.end() ? "x" : "")
    , y_var(_agent.getAgentDescription().variables.find("y") != _agent.getAgentDescription().variables.end() ? "y" : "")
    , z_var(_agent.getAgentDescription().variables.find("z") != _agent.getAgentDescription().variables.end() ? "z" : "")
{ }

AgentStateVis &AgentVis::State(const std::string &state_name) {
    // If state exists
    if (agentData.states.find(state_name) != agentData.states.end()) {
        // If state is not already in vis map
        auto visAgentState = states.find(state_name);
        if (visAgentState == states.end()) {
            // Create new vis agent
            return states.emplace(state_name, AgentStateVis(*this, state_name)).first->second;
        }
        return visAgentState->second;
    }
    THROW InvalidAgentName("State '%s' was not found within agent '%s', "
        "in AgentVis::State()\n",
        state_name.c_str(), agentData.name.c_str());
}


void AgentVis::setXVariable(const std::string &var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setXVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    x_var = var_name;
}
void AgentVis::setYVariable(const std::string &var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setYVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    y_var = var_name;
}
void AgentVis::setZVariable(const std::string &var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setZVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    z_var = var_name;
}
void AgentVis::clearZVariables() {
    z_var = "";
}
std::string AgentVis::getXVariable() const {
    return x_var;
}
std::string AgentVis::getYVariable() const {
    return y_var;
}
std::string AgentVis::getZVariable() const {
    return z_var;
}
void AgentVis::initBindings(std::unique_ptr<FLAMEGPU_Visualisation> &vis) {
    // Pass each state's vis config to the visualiser
    for (auto &state : agentData.states) {
        // For each agent state, give the visualiser
        // vis config
        AgentStateConfig vc = this->defaultConfig;  // Default to parent if child hasn't been configured
        if (states.find(state) != states.end()) {
            vc = states.at(state).config;
        }
        vis->addAgentState(agentData.name, state, vc);
    }
}
void AgentVis::requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation> &vis) {
    for (auto &state : agentData.states) {
        auto &state_map = agent.state_map.at(state);
        vis->requestBufferResizes(agentData.name, state, state_map->getCUDAStateListSize());
    }
}
void AgentVis::updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation> &vis) {
    for (auto &state : agentData.states) {
        auto &state_map = agent.state_map.at(state);
        vis->updateAgentStateBuffer(agentData.name, state,
            state_map->getCUDAStateListSize(),
            reinterpret_cast<float *>(state_map->getAgentListVariablePointer(x_var)),
            reinterpret_cast<float *>(state_map->getAgentListVariablePointer(y_var)),
            reinterpret_cast<float *>(z_var == "" ? nullptr : state_map->getAgentListVariablePointer(z_var)));
    }
    // TODO Tertiary buffers? (e.g. color, direction[xyz])
}

void AgentVis::setModel(const std::string &modelPath, const std::string &texturePath) {
    AgentStateConfig::setString(&defaultConfig.model_path, modelPath);
    if (!texturePath.empty())
        AgentStateConfig::setString(&defaultConfig.model_texture, texturePath);
    // Apply to all states which haven't had the setting overriden
    for (auto &s : states) {
        if (!s.second.configFlags.model_path) {
            AgentStateConfig::setString(&s.second.config.model_path, modelPath);
            if (!texturePath.empty())
                AgentStateConfig::setString(&s.second.config.model_texture, texturePath);
        }
    }
}
void AgentVis::setModel(const Stock::Models::Model &model) {
    AgentStateConfig::setString(&defaultConfig.model_path, model.modelPath);
    if (model.texturePath && model.texturePath[0] != '\0')
        AgentStateConfig::setString(&defaultConfig.model_texture, model.texturePath);
    // Apply to all states which haven't had the setting overridden
    for (auto &s : states) {
        if (!s.second.configFlags.model_path) {
            AgentStateConfig::setString(&s.second.config.model_path, model.modelPath);
            if (model.texturePath && model.texturePath[0] != '\0')
                AgentStateConfig::setString(&s.second.config.model_texture, model.texturePath);
        }
    }
}
void AgentVis::setModelScale(float xLen, float yLen, float zLen) {
    if (xLen <= 0 || yLen <= 0 || zLen <= 0) {
        THROW InvalidArgument("AgentVis::setModelScale(): Invalid argument, lengths must all be positive.\n");
    }
    defaultConfig.model_scale[0] = xLen;
    defaultConfig.model_scale[1] = yLen;
    defaultConfig.model_scale[2] = zLen;
    // Apply to all states which haven't had the setting overriden
    for (auto &s : states) {
        if (!s.second.configFlags.model_scale) {
            s.second.config.model_scale[0] = xLen;
            s.second.config.model_scale[1] = yLen;
            s.second.config.model_scale[2] = zLen;
        }
    }
}
void AgentVis::setModelScale(float maxLen) {
  if (maxLen <= 0) {
        THROW InvalidArgument("AgentVis::setModelScale(): Invalid argument, maxLen must be positive.\n");
    }
    defaultConfig.model_scale[0] = -maxLen;
    // Apply to all states which haven't had the setting overriden
    for (auto &s : states) {
        if (!s.second.configFlags.model_scale) {
            s.second.config.model_scale[0] = -maxLen;
        }
    }
}
