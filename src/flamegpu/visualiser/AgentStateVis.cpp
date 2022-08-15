// @todo - ifdef visualisation?
#include "flamegpu/visualiser/AgentStateVis.h"

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/visualiser/AgentVis.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/visualiser/color/ColorFunction.h"

namespace flamegpu {
namespace visualiser {


AgentStateVis::AgentStateVis(const AgentVis &_parent, const std::string &_state_name)
    : parent(_parent)
    , state_name(_state_name)
    , config(_parent.defaultConfig) { }


void AgentStateVis::setModel(const std::string &modelPath, const std::string &texturePath) {
    AgentStateConfig::setString(&config.model_path, modelPath);
    if (!texturePath.empty())
        AgentStateConfig::setString(&config.model_texture, texturePath);
    configFlags.model_path = 1;
}
void AgentStateVis::setModel(const Stock::Models::Model &model) {
    AgentStateConfig::setString(&config.model_path, model.modelPath);
    configFlags.model_path = 1;
}
void AgentStateVis::setKeyFrameModel(const std::string& modelPathA, const std::string& modelPathB, const std::string& texturePath) {
    if (parent.core_tex_buffers.find(TexBufferConfig::AnimationLerp) == parent.core_tex_buffers.end()) {
        THROW exception::InvalidOperation("Unable to use AgentStateVis::setKeyFrameModel(), AgentVis::setKeyFrameModel()"
            " must be called first to specify the lerp variable for all agent states.\n");
    }
    AgentStateConfig::setString(&config.model_path, modelPathA);
    AgentStateConfig::setString(&config.model_pathB, modelPathB);
    if (!texturePath.empty()) {
        AgentStateConfig::setString(&config.model_texture, texturePath);
        clearColor();
    }
    configFlags.model_path = 1;
}
void AgentStateVis::setKeyFrameModel(const Stock::Models::KeyFrameModel& model) {
    setKeyFrameModel(model.modelPathA, model.modelPathB, model.texturePath ? model.texturePath : "");
}
void AgentStateVis::setModelScale(float xLen, float yLen, float zLen) {
    if (xLen <= 0 || yLen <= 0 || zLen <= 0) {
        THROW exception::InvalidArgument("AgentStateVis::setModelScale(): Invalid argument, lengths must all be positive.\n");
    }
    config.model_scale[0] = xLen;
    config.model_scale[1] = yLen;
    config.model_scale[2] = zLen;
    configFlags.model_scale = 1;
}
void AgentStateVis::setModelScale(float maxLen) {
    if (maxLen <= 0) {
        THROW exception::InvalidArgument("AgentStateVis::setModelScale(): Invalid argument, maxLen must be positive.\n");
    }
    config.model_scale[0] = -maxLen;
    configFlags.model_scale = 1;
}
void AgentStateVis::setColor(const ColorFunction& cf) {
    // Validate agent variable exists
    unsigned int elements = 0;
    if (!cf.getAgentVariableName().empty()) {
        const auto it = parent.agentData.variables.find(cf.getAgentVariableName());
        if (it == parent.agentData.variables.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' bound to color function was not found within agent '%s', "
                "in AgentStateVis::setColor()\n",
                cf.getAgentVariableName().c_str(), parent.agentData.name.c_str());
        }
        if (it->second.type != cf.getAgentVariableRequiredType() || it->second.elements <= cf.getAgentArrayVariableElement()) {
            THROW exception::InvalidAgentVar("Visualisation color function variable must be type %s[>%u], agent '%s' variable '%s' is type %s[%u], "
                "in AgentStateVis::setColor()\n",
                cf.getAgentVariableRequiredType().name(), cf.getAgentArrayVariableElement(), parent.agentData.name.c_str(), cf.getAgentVariableName().c_str(), it->second.type.name(), it->second.elements);
        }
        elements = it->second.elements;
    }
    // Remove old, we only ever want 1 color value
    config.tex_buffers.erase(TexBufferConfig::Color);
    if (!cf.getAgentVariableName().empty() && !cf.getSamplerName().empty())
        config.tex_buffers.emplace(TexBufferConfig::Color, CustomTexBufferConfig{ cf.getAgentVariableName(), cf.getSamplerName(), cf.getAgentArrayVariableElement(), elements });
    config.color_shader_src = cf.getSrc(elements);
}
void AgentStateVis::clearColor() {
    config.tex_buffers.erase(TexBufferConfig::Color);
    config.color_shader_src = "";
}

}  // namespace visualiser
}  // namespace flamegpu
