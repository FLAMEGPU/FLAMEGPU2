// @todo - ifdef visualisation?
#include "flamegpu/visualiser/AgentStateVis.h"

#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/visualiser/AgentVis.h"
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
    // Remove old, we only ever want 1 color value
    config.tex_buffers.erase(TexBufferConfig::Color);
    config.tex_buffers.emplace(TexBufferConfig::Color, CustomTexBufferConfig{ cf.getAgentVariableName(), cf.getSamplerName() });
    config.color_shader_src = cf.getSrc();
}
void AgentStateVis::clearColor() {
    config.tex_buffers.erase(TexBufferConfig::Color);
    config.color_shader_src = "";
}

}  // namespace visualiser
}  // namespace flamegpu
