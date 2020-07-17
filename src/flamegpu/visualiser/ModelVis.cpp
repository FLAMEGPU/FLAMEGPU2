#include "flamegpu/visualiser/ModelVis.h"

#include "flamegpu/gpu/CUDAAgentModel.h"
#include "flamegpu/model/AgentData.h"

ModelVis::ModelVis(const CUDAAgentModel &_model)
    : modelCfg(_model.getModelDescription().name.c_str())
    , model(_model)
    , modelData(_model.getModelDescription()) { }

AgentVis &ModelVis::addAgent(const std::string &agent_name) {
    // If agent exists
    if (modelData.agents.find(agent_name) != modelData.agents.end()) {
        // If agent is not already in vis map
        auto visAgent = agents.find(agent_name);
        if (visAgent == agents.end()) {
            // Create new vis agent
            return agents.emplace(agent_name, AgentVis(model.getCUDAAgent(agent_name))).first->second;
        }
        return visAgent->second;
    }
    THROW InvalidAgentName("Agent name '%s' was not found within the model description hierarchy, "
        "in ModelVis::addAgent()\n",
        agent_name.c_str());
}

AgentVis &ModelVis::Agent(const std::string &agent_name) {
    // If agent exists
    if (modelData.agents.find(agent_name) != modelData.agents.end()) {
        // If agent is not already in vis map
        auto visAgent = agents.find(agent_name);
        if (visAgent != agents.end()) {
            // Create new vis agent
            return visAgent->second;
        }
        THROW InvalidAgentName("Agent name '%s' has not been marked for visualisation, ModelVis::addAgent() must be called first, "
            "in ModelVis::Agent()\n",
            agent_name.c_str());
    }
    THROW InvalidAgentName("Agent name '%s' was not found within the model description hierarchy, "
        "in ModelVis::Agent()\n",
        agent_name.c_str());
}

// Below methods are related to executing the visualiser
void ModelVis::activate() {
    // Only execute if background thread is not active
    if ((!visualiser || !visualiser->isRunning()) && !model.getSimulationConfig().console_mode) {
        // Init visualiser
        visualiser = std::make_unique<FLAMEGPU_Visualisation>(modelCfg);  // Window resolution
        for (auto &agent : agents) {
            // If x and y aren't set, throw exception
            if (agent.second.x_var.empty() || agent.second.y_var.empty()) {
                THROW VisualisationException("Agent '%s' has not had x and y variables set, "
                    "in ModelVis::activate()\n",
                    agent.second.agentData.name.c_str());
            }
            agent.second.initBindings(visualiser);
        }
        visualiser->start();
    }
}

void ModelVis::deactivate() {
    if (visualiser && visualiser->isRunning()) {
        visualiser->stop();
        join();
        visualiser.reset();
    }
}

void ModelVis::join() {
    if (visualiser) {
        visualiser->join();
        visualiser.reset();
    }
}

bool ModelVis::isRunning() const {
    return visualiser ? visualiser->isRunning() : false;
}

void ModelVis::updateBuffers(const unsigned int &sc) {
    if (visualiser) {
        for (auto &a : agents) {
            a.second.requestBufferResizes(visualiser);
        }
        // wait for lock visualiser (its probably executing render loop in separate thread) This might not be 100% safe. RequestResize might need extra thread safety.
        visualiser->lockMutex();
        if (sc != UINT_MAX) {
            visualiser->setStepCount(sc);
        }
        for (auto &a : agents) {
            a.second.updateBuffers(visualiser);
        }
        visualiser->releaseMutex();
    }
}

void ModelVis::setWindowTitle(const std::string& title) {
    ModelConfig::setString(&modelCfg.windowTitle, title);
}

void ModelVis::setWindowDimensions(const unsigned int& width, const unsigned int& height) {
    modelCfg.windowDimensions[0] = width;
    modelCfg.windowDimensions[1] = height;
}

void ModelVis::setClearColor(const float& red, const float& green, const float& blue) {
    modelCfg.clearColor[0] = red;
    modelCfg.clearColor[1] = green;
    modelCfg.clearColor[2] = blue;
}

void ModelVis::setFPSVisible(const bool& showFPS) {
    modelCfg.fpsVisible = showFPS;
}

void ModelVis::setFPSColor(const float& red, const float& green, const float& blue) {
    modelCfg.fpsColor[0] = red;
    modelCfg.fpsColor[1] = green;
    modelCfg.fpsColor[2] = blue;
}

void ModelVis::setInitialCameraLocation(const float &x, const float &y, const float &z) {
    modelCfg.cameraLocation[0] = x;
    modelCfg.cameraLocation[1] = y;
    modelCfg.cameraLocation[2] = z;
}

void ModelVis::setInitialCameraTarget(const float &x, const float &y, const float &z) {
    modelCfg.cameraTarget[0] = x;
    modelCfg.cameraTarget[1] = y;
    modelCfg.cameraTarget[2] = z;
}

void ModelVis::setCameraSpeed(const float &speed, const float &shiftMultiplier) {
    modelCfg.cameraSpeed[0] = speed;
    modelCfg.cameraSpeed[1] = shiftMultiplier;
}

void ModelVis::setViewClips(const float &nearClip, const float &farClip) {
    modelCfg.nearFarClip[0] = nearClip;
    modelCfg.nearFarClip[1] = farClip;
}

void ModelVis::setStepVisible(const bool& showStep) {
    modelCfg.stepVisible = showStep;
}

StaticModelVis ModelVis::addStaticModel(const std::string &modelPath, const std::string &texturePath) {
    // Create ModelConfig::StaticModel
    auto m = std::make_shared<ModelConfig::StaticModel>();
    // set modelPath, texturePath
    m->path = modelPath;
    m->texture = texturePath;
    // add to ModelConfig.staticModels
    modelCfg.staticModels.push_back(m);
    // Create return type
    return StaticModelVis(modelCfg.staticModels.back());
}

LineVis ModelVis::newLineSketch(float r, float g, float b, float a) {
    auto m = std::make_shared<LineConfig>(LineConfig::Type::Lines);
    modelCfg.lines.push_back(m);
    return LineVis(m, r, g, b, a);
}

LineVis ModelVis::newPolylineSketch(float r, float g, float b, float a) {
    auto m = std::make_shared<LineConfig>(LineConfig::Type::Polyline);
    modelCfg.lines.push_back(m);
    return LineVis(m, r, g, b, a);
}
