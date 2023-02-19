// @todo - ifdef visualisation?
#include "flamegpu/visualiser/AgentStateVis.h"

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/visualiser/AgentVis.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/visualiser/color/ColorFunction.h"

namespace flamegpu {
namespace visualiser {


AgentStateVisData::AgentStateVisData(std::shared_ptr<const AgentVisData> _parent, const std::string &_state_name)
    : parent(_parent)
    , state_name(_state_name)
    , config(_parent->defaultConfig) { }


AgentStateVis::AgentStateVis(std::shared_ptr<AgentStateVisData> _data)
    :data(_data) { }
void AgentStateVis::setModel(const std::string &modelPath, const std::string &texturePath) {
    AgentStateConfig::setString(&data->config.model_path, modelPath);
    if (!texturePath.empty())
        AgentStateConfig::setString(&data->config.model_texture, texturePath);
    data->configFlags.model_path = 1;
}
void AgentStateVis::setModel(const Stock::Models::Model &model) {
    AgentStateConfig::setString(&data->config.model_path, model.modelPath);
    data->configFlags.model_path = 1;
}
void AgentStateVis::setKeyFrameModel(const std::string& modelPathA, const std::string& modelPathB, const std::string& texturePath) {
    if (data->parent->core_tex_buffers.find(TexBufferConfig::AnimationLerp) == data->parent->core_tex_buffers.end()) {
        THROW exception::InvalidOperation("Unable to use AgentStateVis::setKeyFrameModel(), AgentVis::setKeyFrameModel()"
            " must be called first to specify the lerp variable for all agent states.\n");
    }
    AgentStateConfig::setString(&data->config.model_path, modelPathA);
    AgentStateConfig::setString(&data->config.model_pathB, modelPathB);
    if (!texturePath.empty()) {
        AgentStateConfig::setString(&data->config.model_texture, texturePath);
        clearColor();
    }
    data->configFlags.model_path = 1;
}
void AgentStateVis::setKeyFrameModel(const Stock::Models::KeyFrameModel& model) {
    setKeyFrameModel(model.modelPathA, model.modelPathB, model.texturePath ? model.texturePath : "");
}
void AgentStateVis::setModelScale(float xLen, float yLen, float zLen) {
    if (xLen <= 0 || yLen <= 0 || zLen <= 0) {
        THROW exception::InvalidArgument("AgentStateVis::setModelScale(): Invalid argument, lengths must all be positive.\n");
    }
    data->config.model_scale[0] = xLen;
    data->config.model_scale[1] = yLen;
    data->config.model_scale[2] = zLen;
    data->configFlags.model_scale = 1;
}
void AgentStateVis::setModelScale(float maxLen) {
    if (maxLen <= 0) {
        THROW exception::InvalidArgument("AgentStateVis::setModelScale(): Invalid argument, maxLen must be positive.\n");
    }
    data->config.model_scale[0] = -maxLen;
    data->configFlags.model_scale = 1;
}
void AgentStateVis::setColor(const ColorFunction& cf) {
    // Validate agent variable exists
    unsigned int elements = 0;
    if (!cf.getAgentVariableName().empty()) {
        const auto it = data->parent->agentData->variables.find(cf.getAgentVariableName());
        if (it == data->parent->agentData->variables.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' bound to color function was not found within agent '%s', "
                "in AgentStateVis::setColor()\n",
                cf.getAgentVariableName().c_str(), data->parent->agentData->name.c_str());
        }
        if (it->second.type != cf.getAgentVariableRequiredType() || it->second.elements <= cf.getAgentArrayVariableElement()) {
            THROW exception::InvalidAgentVar("Visualisation color function variable must be type %s[>%u], agent '%s' variable '%s' is type %s[%u], "
                "in AgentStateVis::setColor()\n",
                cf.getAgentVariableRequiredType().name(), cf.getAgentArrayVariableElement(), data->parent->agentData->name.c_str(), cf.getAgentVariableName().c_str(), it->second.type.name(), it->second.elements);
        }
        elements = it->second.elements;
    }
    // Remove old, we only ever want 1 color value
    data->config.tex_buffers.erase(TexBufferConfig::Color);
    if (!cf.getAgentVariableName().empty() && !cf.getSamplerName().empty())
        data->config.tex_buffers.emplace(TexBufferConfig::Color, CustomTexBufferConfig{ cf.getAgentVariableName(), cf.getSamplerName(), cf.getAgentArrayVariableElement(), elements });
    data->config.color_shader_src = cf.getSrc(elements);
}
void AgentStateVis::clearColor() {
    data->config.tex_buffers.erase(TexBufferConfig::Color);
    data->config.color_shader_src = "";
}
void AgentStateVis::setVisible(bool isVisible) {
    data->visible = isVisible;
}

}  // namespace visualiser
}  // namespace flamegpu
