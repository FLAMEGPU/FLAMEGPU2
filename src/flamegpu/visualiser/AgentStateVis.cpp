#include "flamegpu/visualiser/AgentStateVis.h"

#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/visualiser/AgentVis.h"


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
        THROW InvalidArgument("AgentStateVis::setModelScale(): Invalid argument, lengths must all be positive.\n");
    }
    config.model_scale[0] = xLen;
    config.model_scale[1] = yLen;
    config.model_scale[2] = zLen;
    configFlags.model_scale = 1;
}
void AgentStateVis::setModelScale(float maxLen) {
    if (maxLen <= 0) {
        THROW InvalidArgument("AgentStateVis::setModelScale(): Invalid argument, maxLen must be positive.\n");
    }
    config.model_scale[0] = -maxLen;
    configFlags.model_scale = 1;
}
