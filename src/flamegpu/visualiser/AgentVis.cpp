#include "flamegpu/visualiser/AgentVis.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/visualiser/color/ColorFunction.h"
#include "flamegpu/visualiser/color/StaticColor.h"
#include "FLAMEGPU_Visualisation.h"

AgentVis::AgentVis(CUDAAgent &_agent, const std::shared_ptr<AutoPalette>& autopalette)
    : owned_auto_palette(nullptr)
    , agent(_agent)
    , agentData(_agent.getAgentDescription()) {
    if (_agent.getAgentDescription().variables.find("x") != _agent.getAgentDescription().variables.end()) {
        core_tex_buffers.emplace(TexBufferConfig::Position_x, "x");
    }
    if (_agent.getAgentDescription().variables.find("y") != _agent.getAgentDescription().variables.end()) {
        core_tex_buffers.emplace(TexBufferConfig::Position_y, "y");
    }
    if (_agent.getAgentDescription().variables.find("z") != _agent.getAgentDescription().variables.end()) {
        core_tex_buffers.emplace(TexBufferConfig::Position_z, "z");
    }
    if (autopalette) {
        setColor(autopalette->next());
        auto_palette = autopalette;  // setColor() clears auto_palette
    }
}

AgentStateVis &AgentVis::State(const std::string &state_name) {
    // If state exists
    if (agentData.states.find(state_name) != agentData.states.end()) {
        // If state is not already in vis map
        auto visAgentState = states.find(state_name);
        if (visAgentState == states.end()) {
            // Create new vis agent
            auto &rtn = states.emplace(state_name, AgentStateVis(*this, state_name)).first->second;
            auto ap = auto_palette.lock();
            if (ap) {
                const Color c = ap->next();
                rtn.setColor(c);
            }
            return rtn;
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
    core_tex_buffers[TexBufferConfig::Position_x].agentVariableName = var_name;
}
void AgentVis::setYVariable(const std::string &var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setYVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Position_y].agentVariableName = var_name;
}
void AgentVis::setZVariable(const std::string &var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setZVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Position_z].agentVariableName = var_name;
}
void AgentVis::setForwardXVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setDirectionXVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Forward_x].agentVariableName = var_name;
}
void AgentVis::setForwardYVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setDirectionYVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Forward_y].agentVariableName = var_name;
}
void AgentVis::setForwardZVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setDirectionZVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Forward_z].agentVariableName = var_name;
}
void AgentVis::setUpXVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpXVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Up_x].agentVariableName = var_name;
}
void AgentVis::setUpYVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpYVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Up_y].agentVariableName = var_name;
}
void AgentVis::setUpZVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setUpZVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Up_z].agentVariableName = var_name;
}
void AgentVis::setYawVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setYawVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Heading].agentVariableName = var_name;
}
void AgentVis::setPitchVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setPitchVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Pitch].agentVariableName = var_name;
}
void AgentVis::setRollVariable(const std::string& var_name) {
    if (agentData.variables.find(var_name) == agentData.variables.end()) {
        THROW InvalidAgentVar("Variable '%s' was not found within agent '%s', "
            "in AgentVis::setRollVariable()\n",
            var_name.c_str(), agentData.name.c_str());
    }
    core_tex_buffers[TexBufferConfig::Bank].agentVariableName = var_name;
}
void AgentVis::clearXVariable() {
    core_tex_buffers.erase(TexBufferConfig::Position_x);
}
void AgentVis::clearYVariable() {
    core_tex_buffers.erase(TexBufferConfig::Position_y);
}
void AgentVis::clearZVariable() {
    core_tex_buffers.erase(TexBufferConfig::Position_z);
}
void AgentVis::clearForwardXVariable() {
    core_tex_buffers.erase(TexBufferConfig::Forward_x);
}
void AgentVis::clearForwardYVariable() {
    core_tex_buffers.erase(TexBufferConfig::Forward_y);
}
void AgentVis::clearForwardZVariable() {
    core_tex_buffers.erase(TexBufferConfig::Forward_z);
}
void AgentVis::clearUpXVariable() {
    core_tex_buffers.erase(TexBufferConfig::Up_x);
}
void AgentVis::clearUpYVariable() {
    core_tex_buffers.erase(TexBufferConfig::Up_y);
}
void AgentVis::clearUpZVariable() {
    core_tex_buffers.erase(TexBufferConfig::Up_z);
}
void AgentVis::clearYawVariable() {
    core_tex_buffers.erase(TexBufferConfig::Heading);
}
void AgentVis::clearPitchVariable() {
    core_tex_buffers.erase(TexBufferConfig::Pitch);
}
void AgentVis::clearRollVariable() {
    core_tex_buffers.erase(TexBufferConfig::Bank);
}
std::string AgentVis::getXVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Position_x);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getYVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Position_y);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getZVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Position_z);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardXVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Forward_x);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardYVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Forward_y);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getForwardZVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Forward_z);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpXVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Up_x);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpYVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Up_y);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getUpZVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Up_z);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getYawVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Heading);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getPitchVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Pitch);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
std::string AgentVis::getRollVariable() const {
    const auto it = core_tex_buffers.find(TexBufferConfig::Bank);
    return it != core_tex_buffers.end() ? it->second.agentVariableName : "";
}
void AgentVis::initBindings(std::unique_ptr<FLAMEGPU_Visualisation> &vis) {
    // Pass each state's vis config to the visualiser
    for (auto &state : agentData.states) {
        // For each agent state, give the visualiser
        // vis config
        AgentStateConfig &vc = this->defaultConfig;  // Default to parent if child hasn't been configured
        if (states.find(state) != states.end()) {
            vc = states.at(state).config;
        }
        vis->addAgentState(agentData.name, state, vc, core_tex_buffers, vc.tex_buffers);
    }
}
bool AgentVis::requestBufferResizes(std::unique_ptr<FLAMEGPU_Visualisation> &vis, bool force) {
    unsigned int agents_requested = 0;
    for (auto &state : agentData.states) {
        auto &state_map = agent.state_map.at(state);
        vis->requestBufferResizes(agentData.name, state, state_map->getSize(), force);
        agents_requested += state_map->getSize();
    }
    return agents_requested;
}
void AgentVis::updateBuffers(std::unique_ptr<FLAMEGPU_Visualisation> &vis) {
    for (auto &state : agentData.states) {
        AgentStateConfig &state_config = this->defaultConfig;  // Default to parent if child hasn't been configured
        if (states.find(state) != states.end()) {
            state_config = states.at(state).config;
        }
        auto& state_data_map = agent.state_map.at(state);
        // Update buffer pointers inside the map
        // These get changed per state, but should be fine
        for (auto &tb : core_tex_buffers) {
            tb.second.t_d_ptr = state_data_map->getVariablePointer(tb.second.agentVariableName);
        }
        for (auto &tb : state_config.tex_buffers) {
            tb.second.t_d_ptr = state_data_map->getVariablePointer(tb.second.agentVariableName);
        }
        // Pass the updated map to the update function
        vis->updateAgentStateBuffer(agentData.name, state, state_data_map->getSize(), core_tex_buffers, state_config.tex_buffers);
    }
}

void AgentVis::setModel(const std::string &modelPath, const std::string &texturePath) {
    AgentStateConfig::setString(&defaultConfig.model_path, modelPath);
    if (!texturePath.empty()) {
        AgentStateConfig::setString(&defaultConfig.model_texture, texturePath);
        clearColor();
    }
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

void AgentVis::setAutoPalette(const Palette& ap) {
    owned_auto_palette = std::make_shared<AutoPalette>(ap);
    auto_palette = owned_auto_palette;
}
void AgentVis::setColor(const ColorFunction& cf) {
    // Remove old, we only ever want 1 color value
    defaultConfig.tex_buffers.erase(TexBufferConfig::Color);
    if (!cf.getAgentVariableName().empty() && !cf.getSamplerName().empty())
        defaultConfig.tex_buffers.emplace(TexBufferConfig::Color, CustomTexBufferConfig{ cf.getAgentVariableName(), cf.getSamplerName() });
    defaultConfig.color_shader_src = cf.getSrc();
    auto_palette.reset();
    owned_auto_palette = nullptr;
    // Clear texture, can't have both colour and texture
    if (defaultConfig.model_texture) {
        free(const_cast<char*>(defaultConfig.model_texture));
    }
}
void AgentVis::clearColor() {
    defaultConfig.tex_buffers.erase(TexBufferConfig::Color);
    defaultConfig.color_shader_src = "";
    auto_palette.reset();
    owned_auto_palette = nullptr;
}
