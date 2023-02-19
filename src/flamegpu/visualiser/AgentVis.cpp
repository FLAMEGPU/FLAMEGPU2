// @todo - ifdef visualisation?

#include "flamegpu/visualiser/AgentVis.h"

#include "flamegpu/simulation/detail/CUDAAgent.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/visualiser/color/ColorFunction.h"
#include "flamegpu/visualiser/color/StaticColor.h"
#include "flamegpu/visualiser/color/AutoPalette.h"
#include "flamegpu/visualiser/FLAMEGPU_Visualisation.h"

namespace flamegpu {
namespace visualiser {

AgentVisData::AgentVisData(detail::CUDAAgent &_agent, const std::shared_ptr<AutoPalette>& autopalette)
    : owned_auto_palette(nullptr)
    , agent(_agent)
    , agentData(std::const_pointer_cast<const AgentData>(_agent.getAgentDescription().agent)) {
    const CAgentDescription agent_desc = _agent.getAgentDescription();
    if (agent_desc.hasVariable("x") &&
        agent_desc.getVariableType("x") == std::type_index(typeid(float)) &&
        agent_desc.getVariableLength("x") == 1) {
        core_tex_buffers[TexBufferConfig::Position_x].agentVariableName = "x";
    }
    if (agent_desc.hasVariable("y") &&
        agent_desc.getVariableType("y") == std::type_index(typeid(float)) &&
        agent_desc.getVariableLength("y") == 1) {
        core_tex_buffers[TexBufferConfig::Position_y].agentVariableName = "y";
    }
    if (agent_desc.hasVariable("z") &&
        agent_desc.getVariableType("z") == std::type_index(typeid(float)) &&
        agent_desc.getVariableLength("z") == 1) {
        core_tex_buffers[TexBufferConfig::Position_z].agentVariableName = "z";
    }
    if (autopalette) {
        defaultConfig.color_shader_src = StaticColor(autopalette->next()).getSrc(0);  // Arg is not used by static color, so 0 can be passed
    }
}
void AgentVisData::initBindings(std::unique_ptr<FLAMEGPU_Visualisation>& vis) {
    // Pass each state's vis config to the visualiser
    for (auto& state : agentData->states) {
        // For each agent state, give the visualiser
        // vis config
        AgentStateConfig& vc = defaultConfig;  // Default to parent if child hasn't been configured
        const auto state_it = states.find(state);
        if (state_it != states.end()) {
            if (!state_it->second->visible) {
                // Skip agent states marked hidden
                continue;
            }
            vc = state_it->second->config;
        }
        vis->addAgentState(agentData->name, state, vc, core_tex_buffers, vc.tex_buffers);
    }
}
bool AgentVisData::requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation>& vis, bool force) {
    unsigned int agents_requested = 0;
    for (auto& state : agentData->states) {
        const auto state_it = states.find(state);
        if (state_it != states.end()) {
            if (!state_it->second->visible) {
                // Skip agent states marked hidden
                continue;
            }
        }
        auto& state_map = agent.state_map.at(state);
        vis->requestBufferResizes(agentData->name, state, state_map->getSize(), force);
        agents_requested += state_map->getSize();
    }
    return agents_requested;
}
void AgentVisData::updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation>& vis) {
    for (auto& state : agentData->states) {
        AgentStateConfig& state_config = defaultConfig;  // Default to parent if child hasn't been configured
        const auto state_it = states.find(state);
        if (state_it != states.end()) {
            if (!state_it->second->visible) {
                // Skip agent states marked hidden
                continue;
            }
            state_config = state_it->second->config;
        }
        auto& state_data_map = agent.state_map.at(state);
        // Update buffer pointers inside the map
        // These get changed per state, but should be fine
        for (auto& tb : core_tex_buffers) {
            tb.second.t_d_ptr = state_data_map->getVariablePointer(tb.second.agentVariableName);
        }
        for (auto& tb : state_config.tex_buffers) {
            tb.second.t_d_ptr = state_data_map->getVariablePointer(tb.second.agentVariableName);
        }
        // Pass the updated map to the update function
        vis->updateAgentStateBuffer(agentData->name, state, state_data_map->getSize(), core_tex_buffers, state_config.tex_buffers);
    }
}
AgentVis::AgentVis(std::shared_ptr<AgentVisData> _data)
    : data(std::move(_data)) { }

AgentStateVis AgentVis::State(const std::string &state_name) {
    // If state exists
    if (data->agentData->states.find(state_name) != data->agentData->states.end()) {
        // If state is not already in vis map
        auto visAgentState = data->states.find(state_name);
        if (visAgentState == data->states.end()) {
            // Create new vis agent
            auto rtn = data->states.emplace(state_name, std::make_shared<AgentStateVisData>(std::const_pointer_cast<const AgentVisData>(data), state_name)).first->second;
            auto ap = data->auto_palette.lock();
            if (ap) {
                rtn->config.color_shader_src = StaticColor(ap->next()).getSrc(0);  // Arg is not used by static color, so 0 can be passed
            }
            return AgentStateVis(rtn);
        }
        return AgentStateVis(visAgentState->second);
    }
    THROW exception::InvalidAgentName("State '%s' was not found within agent '%s', "
        "in AgentVis::State()\n",
        state_name.c_str(), data->agentData->name.c_str());
}


void AgentVis::setXVariable(const std::string &var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setXVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation position x variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setXVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Position_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Position_xyz);
    data->core_tex_buffers[TexBufferConfig::Position_x].agentVariableName = var_name;
}
void AgentVis::setYVariable(const std::string &var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setYVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation position Y variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setYVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Position_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Position_xyz);
    data->core_tex_buffers[TexBufferConfig::Position_y].agentVariableName = var_name;
}
void AgentVis::setZVariable(const std::string &var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation position z variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Position_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Position_xyz);
    data->core_tex_buffers[TexBufferConfig::Position_z].agentVariableName = var_name;
}
void AgentVis::setXYVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setXYVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 2) {
        THROW exception::InvalidAgentVar("Visualisation position x variable must be type float[2], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setXYVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Position_x);
    data->core_tex_buffers.erase(TexBufferConfig::Position_y);
    data->core_tex_buffers.erase(TexBufferConfig::Position_z);
    data->core_tex_buffers.erase(TexBufferConfig::Position_xyz);
    data->core_tex_buffers[TexBufferConfig::Position_xy].agentVariableName = var_name;
}
void AgentVis::setXYZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setXYZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 3) {
        THROW exception::InvalidAgentVar("Visualisation position x variable must be type float[3], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setXYZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Position_x);
    data->core_tex_buffers.erase(TexBufferConfig::Position_y);
    data->core_tex_buffers.erase(TexBufferConfig::Position_z);
    data->core_tex_buffers.erase(TexBufferConfig::Position_xy);
    data->core_tex_buffers[TexBufferConfig::Position_xyz].agentVariableName = var_name;
}
void AgentVis::setForwardXVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setForwardXVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation forward x variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setForwardXVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Forward_x].agentVariableName = var_name;
}
void AgentVis::setForwardYVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setForwardYVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation forward y variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setForwardYVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Pitch);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Forward_y].agentVariableName = var_name;
}
void AgentVis::setForwardZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setForwardZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation forward z variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setForwardZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Forward_z].agentVariableName = var_name;
}
void AgentVis::setForwardXZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setForwardXZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 2) {
        THROW exception::InvalidAgentVar("Visualisation forward xz variable must be type float[2], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setForwardXZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_x);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_y);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_z);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Forward_xz].agentVariableName = var_name;
}
void AgentVis::setForwardXYZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setForwardXYZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 3) {
        THROW exception::InvalidAgentVar("Visualisation forward xyz variable must be type float[2], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setForwardXYZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_x);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_z);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Forward_xyz].agentVariableName = var_name;
}
void AgentVis::setUpXVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpXVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation up x variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setUpXVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Bank);
    data->core_tex_buffers.erase(TexBufferConfig::Up_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Up_x].agentVariableName = var_name;
}
void AgentVis::setUpYVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpYVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation up y variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setUpYVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Bank);
    data->core_tex_buffers.erase(TexBufferConfig::Up_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Up_y].agentVariableName = var_name;
}
void AgentVis::setUpZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation up z variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setUpZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Bank);
    data->core_tex_buffers.erase(TexBufferConfig::Up_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Up_z].agentVariableName = var_name;
}
void AgentVis::setUpXYZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpXYZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 3) {
        THROW exception::InvalidAgentVar("Visualisation up xyz variable must be type float[3], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setUpXYZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Bank);
    data->core_tex_buffers.erase(TexBufferConfig::Up_x);
    data->core_tex_buffers.erase(TexBufferConfig::Up_y);
    data->core_tex_buffers.erase(TexBufferConfig::Up_z);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Up_xyz].agentVariableName = var_name;
}
void AgentVis::setYawVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setYawVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation yaw variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setYawVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Forward_x);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_z);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Heading].agentVariableName = var_name;
}
void AgentVis::setPitchVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setPitchVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation pitch variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setPitchVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Forward_y);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Pitch].agentVariableName = var_name;
}
void AgentVis::setRollVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setRollVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation roll variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setRollVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Up_x);
    data->core_tex_buffers.erase(TexBufferConfig::Up_y);
    data->core_tex_buffers.erase(TexBufferConfig::Up_z);
    data->core_tex_buffers.erase(TexBufferConfig::Up_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Bank].agentVariableName = var_name;
}
void AgentVis::setDirectionYPVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setDirectionYPVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 2) {
        THROW exception::InvalidAgentVar("Visualisation direction yaw/pitch variable must be type float[2], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setDirectionYPVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Forward_x);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_z);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
    data->core_tex_buffers.erase(TexBufferConfig::Pitch);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
    data->core_tex_buffers[TexBufferConfig::Heading].agentVariableName = var_name;
}
void AgentVis::setDirectionYPRVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setDirectionYPRVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 3) {
        THROW exception::InvalidAgentVar("Visualisation direction yaw/pitch/roll variable must be type float[3], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setDirectionYPRVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Forward_x);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_z);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Up_x);
    data->core_tex_buffers.erase(TexBufferConfig::Up_y);
    data->core_tex_buffers.erase(TexBufferConfig::Up_z);
    data->core_tex_buffers.erase(TexBufferConfig::Up_xyz);
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
    data->core_tex_buffers.erase(TexBufferConfig::Pitch);
    data->core_tex_buffers.erase(TexBufferConfig::Bank);
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
    data->core_tex_buffers[TexBufferConfig::Heading].agentVariableName = var_name;
}
void AgentVis::setUniformScaleVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUniformScaleVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation scale variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setUniformScaleVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::Scale_x);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_y);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_z);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xyz);
    data->core_tex_buffers[TexBufferConfig::UniformScale].agentVariableName = var_name;
}
void AgentVis::setScaleXVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setScaleXVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation scale x variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setScaleXVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::UniformScale);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xyz);
    data->core_tex_buffers[TexBufferConfig::Scale_x].agentVariableName = var_name;
}
void AgentVis::setScaleYVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setScaleYVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation scale y variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setScaleYVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::UniformScale);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xyz);
    data->core_tex_buffers[TexBufferConfig::Scale_y].agentVariableName = var_name;
}
void AgentVis::setScaleZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setScaleZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation scale z variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setScaleZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::UniformScale);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xy);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xyz);
    data->core_tex_buffers[TexBufferConfig::Scale_z].agentVariableName = var_name;
}
void AgentVis::setScaleXYVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setScaleXYVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 2) {
        THROW exception::InvalidAgentVar("Visualisation scale xy variable must be type float[2], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setScaleXYVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::UniformScale);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_x);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_y);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_z);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xyz);
    data->core_tex_buffers[TexBufferConfig::Scale_xy].agentVariableName = var_name;
}
void AgentVis::setScaleXYZVariable(const std::string& var_name) {
    auto it = data->agentData->variables.find(var_name);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setScaleXYZVariable()\n",
            var_name.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 3) {
        THROW exception::InvalidAgentVar("Visualisation scale xyz variable must be type float[3], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setScaleXYZVariable()\n",
            data->agentData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->core_tex_buffers.erase(TexBufferConfig::UniformScale);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_x);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_y);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_z);
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xy);
    data->core_tex_buffers[TexBufferConfig::Scale_xyz].agentVariableName = var_name;
}
void AgentVis::clearXVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Position_x);
}
void AgentVis::clearYVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Position_y);
}
void AgentVis::clearZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Position_z);
}
void AgentVis::clearXYVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Position_xy);
}
void AgentVis::clearXYZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Position_xyz);
}
void AgentVis::clearForwardXVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Forward_x);
}
void AgentVis::clearForwardYVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Forward_y);
}
void AgentVis::clearForwardZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Forward_z);
}
void AgentVis::clearForwardXZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xz);
}
void AgentVis::clearForwardXYZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Forward_xyz);
}
void AgentVis::clearUpXVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Up_x);
}
void AgentVis::clearUpYVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Up_y);
}
void AgentVis::clearUpZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Up_z);
}
void AgentVis::clearUpXYZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Up_xyz);
}
void AgentVis::clearYawVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Heading);
}
void AgentVis::clearPitchVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Pitch);
}
void AgentVis::clearRollVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Bank);
}
void AgentVis::clearDirectionYPVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hp);
}
void AgentVis::clearDirectionYPRVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Direction_hpb);
}
void AgentVis::clearUniformScaleVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::UniformScale);
}
void AgentVis::clearScaleXVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Scale_x);
}
void AgentVis::clearScaleYVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Scale_y);
}
void AgentVis::clearScaleZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Scale_z);
}
void AgentVis::clearScaleXYVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xy);
}
void AgentVis::clearScaleXYZVariable() {
    data->core_tex_buffers.erase(TexBufferConfig::Scale_xyz);
}
std::string AgentVis::getXVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Position_x);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getYVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Position_y);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Position_z);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getXYVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Position_xy);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getXYZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Position_xyz);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardXVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Forward_x);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardYVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Forward_y);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Forward_z);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpXVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Up_x);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardXZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Forward_xz);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardXYZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Forward_xyz);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpYVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Up_y);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Up_z);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpXYZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Up_xyz);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getYawVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Heading);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getPitchVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Pitch);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getRollVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Bank);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getDirectionYPVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Direction_hp);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getDirectionYPRVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Direction_hpb);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUniformScaleVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::UniformScale);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getScaleXVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Scale_x);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getScaleYVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Scale_y);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getScaleZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Scale_z);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getScaleXYVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Scale_xy);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getScaleXYZVariable() const {
    const auto it = data->core_tex_buffers.find(TexBufferConfig::Scale_xyz);
    return it != data->core_tex_buffers.end() ? it->second.agentVariableName : "";
}

void AgentVis::setModel(const std::string &modelPath, const std::string &texturePath) {
    AgentStateConfig::setString(&data->defaultConfig.model_path, modelPath);
    if (!texturePath.empty()) {
        AgentStateConfig::setString(&data->defaultConfig.model_texture, texturePath);
        clearColor();
    }
    // Apply to all states which haven't had the setting overriden
    for (auto &s : data->states) {
        if (!s.second->configFlags.model_path) {
            AgentStateConfig::setString(&s.second->config.model_path, modelPath);
            if (!texturePath.empty())
                AgentStateConfig::setString(&s.second->config.model_texture, texturePath);
        }
    }
}
void AgentVis::setModel(const Stock::Models::Model &model) {
    AgentStateConfig::setString(&data->defaultConfig.model_path, model.modelPath);
    if (model.texturePath && model.texturePath[0] != '\0') {
        AgentStateConfig::setString(&data->defaultConfig.model_texture, model.texturePath);
        clearColor();
    }
    // Apply to all states which haven't had the setting overridden
    for (auto &s : data->states) {
        if (!s.second->configFlags.model_path) {
            AgentStateConfig::setString(&s.second->config.model_path, model.modelPath);
            if (model.texturePath && model.texturePath[0] != '\0')
                AgentStateConfig::setString(&s.second->config.model_texture, model.texturePath);
        }
    }
}
void AgentVis::setKeyFrameModel(const std::string& modelPathA, const std::string& modelPathB, const std::string& lerpVariableName, const std::string& texturePath) {
    auto it = data->agentData->variables.find(lerpVariableName);
    if (it == data->agentData->variables.end()) {
        THROW exception::InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setKeyFrameModel()\n",
            lerpVariableName.c_str(), data->agentData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidAgentVar("Visualisation animation lerp variable must be type float[1], agent '%s' variable '%s' is type %s[%u], "
            "in AgentVis::setKeyFrameModel()\n",
            data->agentData->name.c_str(), lerpVariableName.c_str(), it->second.type.name(), it->second.elements);
    }
    AgentStateConfig::setString(&data->defaultConfig.model_path, modelPathA);
    AgentStateConfig::setString(&data->defaultConfig.model_pathB, modelPathB);
    if (!texturePath.empty()) {
        AgentStateConfig::setString(&data->defaultConfig.model_texture, texturePath);
        clearColor();
    }
    data->core_tex_buffers.erase(TexBufferConfig::AnimationLerp);
    data->core_tex_buffers[TexBufferConfig::AnimationLerp].agentVariableName = lerpVariableName;
    // Apply to all states which haven't had the setting overridden
    for (auto& s : data->states) {
        if (!s.second->configFlags.model_path) {
            AgentStateConfig::setString(&s.second->config.model_path, modelPathA);
            AgentStateConfig::setString(&s.second->config.model_pathB, modelPathB);
            if (!texturePath.empty()) {
                AgentStateConfig::setString(&s.second->config.model_texture, texturePath);
                // Clear colour in state
                s.second->config.color_shader_src = "";
            }
        }
    }
}
void AgentVis::setKeyFrameModel(const Stock::Models::KeyFrameModel& model, const std::string& lerpVariableName) {
    setKeyFrameModel(model.modelPathA, model.modelPathB, lerpVariableName, model.texturePath ? model.texturePath : "");
}
void AgentVis::setModelScale(float xLen, float yLen, float zLen) {
    if (xLen <= 0 || yLen <= 0 || zLen <= 0) {
        THROW exception::InvalidArgument("AgentVis::setModelScale(): Invalid argument, lengths must all be positive.\n");
    }
    data->defaultConfig.model_scale[0] = xLen;
    data->defaultConfig.model_scale[1] = yLen;
    data->defaultConfig.model_scale[2] = zLen;
    // Apply to all states which haven't had the setting overriden
    for (auto &s : data->states) {
        if (!s.second->configFlags.model_scale) {
            s.second->config.model_scale[0] = xLen;
            s.second->config.model_scale[1] = yLen;
            s.second->config.model_scale[2] = zLen;
        }
    }
}
void AgentVis::setModelScale(float maxLen) {
  if (maxLen <= 0) {
        THROW exception::InvalidArgument("AgentVis::setModelScale(): Invalid argument, maxLen must be positive.\n");
    }
    data->defaultConfig.model_scale[0] = -maxLen;
    // Apply to all states which haven't had the setting overriden
    for (auto &s : data->states) {
        if (!s.second->configFlags.model_scale) {
            s.second->config.model_scale[0] = -maxLen;
        }
    }
}

void AgentVis::setAutoPalette(const Palette& ap) {
    data->owned_auto_palette = std::make_shared<AutoPalette>(ap);
    data->auto_palette = data->owned_auto_palette;
}
void AgentVis::setColor(const ColorFunction& cf) {
    // Validate agent variable exists
    unsigned int elements = 0;
    if (!cf.getAgentVariableName().empty()) {
        const auto it = data->agentData->variables.find(cf.getAgentVariableName());
        if (it == data->agentData->variables.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' bound to color function was not found within agent '%s', "
                "in AgentVis::setColor()\n",
                cf.getAgentVariableName().c_str(), data->agentData->name.c_str());
        }
        if (it->second.type != cf.getAgentVariableRequiredType() || it->second.elements <= cf.getAgentArrayVariableElement()) {
            THROW exception::InvalidAgentVar("Visualisation color function variable must be type %s[>%u], agent '%s' variable '%s' is type %s[%u], "
                "in AgentVis::setColor()\n",
                cf.getAgentVariableRequiredType().name(), cf.getAgentArrayVariableElement(), data->agentData->name.c_str(), cf.getAgentVariableName().c_str(), it->second.type.name(), it->second.elements);
        }
        elements = it->second.elements;
    }
    // Remove old, we only ever want 1 color value
    data->defaultConfig.tex_buffers.erase(TexBufferConfig::Color);
    if (!cf.getAgentVariableName().empty() && !cf.getSamplerName().empty())
        data->defaultConfig.tex_buffers.emplace(TexBufferConfig::Color, CustomTexBufferConfig{ cf.getAgentVariableName(), cf.getSamplerName(), cf.getAgentArrayVariableElement(), elements });
    data->defaultConfig.color_shader_src = cf.getSrc(elements);
    data->auto_palette.reset();
    data->owned_auto_palette = nullptr;
    // Clear texture, can't have both colour and texture
    if (data->defaultConfig.model_texture) {
        free(const_cast<char*>(data->defaultConfig.model_texture));
    }
}
void AgentVis::clearColor() {
    data->defaultConfig.tex_buffers.erase(TexBufferConfig::Color);
    data->defaultConfig.color_shader_src = "";
    data->auto_palette.reset();
    data->owned_auto_palette = nullptr;
}

}  // namespace visualiser
}  // namespace flamegpu
