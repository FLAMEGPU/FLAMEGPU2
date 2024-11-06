// @todo - ifdef visualisation

#include "flamegpu/visualiser/ModelVis.h"

#include <thread>
#include <utility>
#include <string>
#include <memory>
#include <unordered_map>

#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/visualiser/FLAMEGPU_Visualisation.h"

namespace flamegpu {
namespace visualiser {

ModelVisData::ModelVisData(const flamegpu::CUDASimulation &_model)
: modelCfg(_model.getModelDescription().name.c_str())
, autoPalette(std::make_shared<AutoPalette>(Stock::Palettes::DARK2))
, model(_model)
, modelData(_model.getModelDescription()) { }

void ModelVisData::hookVis(std::shared_ptr<visualiser::ModelVisData>& vis, std::unordered_map<std::string, std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>> &map) {
    for (auto [name, graph] : graphs) {
        auto &graph_buffs = map.at(name);
        graph_buffs->setVisualisation(vis);
        graph->constructGraph(graph_buffs);
        vis->rebuildEnvGraph(name);
    }
}
void ModelVisData::registerEnvProperties() {
    if (model.singletons && !env_registered) {
        char* const host_env_origin = const_cast<char*>(static_cast<const char*>(model.singletons->environment->getHostBuffer()));
        for (const auto &panel : modelCfg.panels) {
            for (const auto &element : panel.second->ui_elements) {
                if (auto a = dynamic_cast<EnvPropertyElement*>(element.get())) {
                    auto & prop = model.singletons->environment->getPropertiesMap().at(a->getName());
                    visualiser->registerEnvironmentProperty(a->getName(), host_env_origin + prop.offset, prop.type, prop.elements, prop.isConst);
                }
            }
        }
        env_registered = true;
    }
}
void ModelVisData::updateBuffers(const unsigned int& sc) {
    if (visualiser) {
        bool has_agents = false;
        for (auto& a : agents) {
            has_agents = a.second->requestBufferResizes(visualiser, sc == 0 || sc == UINT_MAX) || has_agents;
        }
        // Block the sim when we first get agents, until vis has resized buffers, incase vis is being slow to init
        if (has_agents && (sc == 0 || sc == UINT_MAX)) {
            while (!visualiser->buffersReady()) {
                // Do nothing, just spin until ready
                std::this_thread::yield();
            }
        }
        // wait for lock data->visualiser (its probably executing render loop in separate thread) This might not be 100% safe. RequestResize might need extra thread safety.
        visualiser->lockMutex();
        // Update step count
        if (sc != UINT_MAX) {
            visualiser->setStepCount(sc);
        }
        for (auto& a : agents) {
            a.second->updateBuffers(visualiser);
        }
        visualiser->releaseMutex();
        // Block the sim again, until vis is fully ready
        if (has_agents && (sc == 0 || sc == UINT_MAX)) {
            while (!visualiser->isReady()) {
                // Do nothing, just spin until ready
                std::this_thread::yield();
            }
        }
    }
}
void ModelVisData::updateRandomSeed() {
    if (visualiser) {
        // Yolo thread safety, shouldn't matter if random seed is printed wrong for a single frame
        visualiser->setRandomSeed(model.getSimulationConfig().random_seed);
    }
}
void ModelVisData::buildEnvGraphs() {
    for (auto [name, graph] : graphs) {
        graph->constructGraph(model.directed_graph_map.at(name));
    }
}
void ModelVisData::rebuildEnvGraph(const std::string &graph_name) {
    graphs.at(graph_name)->constructGraph(model.directed_graph_map.at(graph_name));
}

ModelVis::ModelVis(std::shared_ptr<ModelVisData> _data, bool _isSWIG)
    : isSWIG(_isSWIG)
    , data(std::move(_data)) { }

void ModelVis::setAutoPalette(const Palette& palette) {
    data->autoPalette = std::make_shared<AutoPalette>(palette);
}
void ModelVis::clearAutoPalette() {
    data->autoPalette = nullptr;
}
AgentVis ModelVis::addAgent(const std::string &agent_name) {
    // If agent exists
    if (data->modelData.agents.find(agent_name) != data->modelData.agents.end()) {
        // If agent is not already in vis map
        auto visAgent = data->agents.find(agent_name);
        if (visAgent == data->agents.end()) {
            // Create new vis agent
            return AgentVis(data->agents.emplace(agent_name, std::make_shared<AgentVisData>(data->model.getCUDAAgent(agent_name), data->autoPalette)).first->second);
        }
        return AgentVis(visAgent->second);
    }
    THROW exception::InvalidAgentName("Agent name '%s' was not found within the model description hierarchy, "
        "in ModelVis::addAgent()\n",
        agent_name.c_str());
}

AgentVis ModelVis::Agent(const std::string &agent_name) {
    // If agent exists
    if (data->modelData.agents.find(agent_name) != data->modelData.agents.end()) {
        // If agent is already in vis map
        auto visAgent = data->agents.find(agent_name);
        if (visAgent != data->agents.end()) {
            // Create new vis agent
            return AgentVis(visAgent->second);
        }
        THROW exception::InvalidAgentName("Agent name '%s' has not been marked for visualisation, ModelVis::addAgent() must be called first, "
            "in ModelVis::Agent()\n",
            agent_name.c_str());
    }
    THROW exception::InvalidAgentName("Agent name '%s' was not found within the model description hierarchy, "
        "in ModelVis::Agent()\n",
        agent_name.c_str());
}
EnvironmentGraphVis ModelVis::addGraph(const std::string& graph_name) {
    // If graph exists
    auto graph_it = data->modelData.environment->directed_graphs.find(graph_name);
    if (graph_it != data->modelData.environment->directed_graphs.end()) {
        // If graph is not already in vis map
        auto visGraph = data->graphs.find(graph_name);
        if (visGraph == data->graphs.end()) {
            // Create a line config for the graph
            auto m = std::make_shared<LineConfig>(LineConfig::Type::Lines);
            data->modelCfg.dynamic_lines.insert({std::string("graph_") + graph_name, m});
            // Create new vis graph
            return EnvironmentGraphVis(data->graphs.emplace(graph_name, std::make_shared<EnvironmentGraphVisData>(graph_it->second, m)).first->second);
        }
        return EnvironmentGraphVis(visGraph->second);
    }
    THROW exception::InvalidEnvGraph("Environment direct graph name '%s' was not found within the model description hierarchy, "
        "in ModelVis::addGraph()\n",
        graph_name.c_str());
}
EnvironmentGraphVis ModelVis::Graph(const std::string& graph_name) {
    // If graph exists
    if (data->modelData.environment->directed_graphs.find(graph_name) != data->modelData.environment->directed_graphs.end()) {
        // If graph is already in vis map
        auto visGraph = data->graphs.find(graph_name);
        if (visGraph != data->graphs.end()) {
            return EnvironmentGraphVis(visGraph->second);
        }
        THROW exception::InvalidEnvGraph("Environment direct graph name '%s' has not been marked for visualisation, ModelVis::addGraph() must be called first, "
            "in ModelVis::Agent()\n",
            graph_name.c_str());
    }
    THROW exception::InvalidEnvGraph("Environment direct graph name '%s' was not found within the model description hierarchy, "
        "in ModelVis::addGraph()\n",
        graph_name.c_str());
}
// Below methods are related to executing the visualiser
void ModelVis::activate() {
    // Only execute if background thread is not active
    if ((!data->visualiser || !data->visualiser->isRunning()) && !data->model.getSimulationConfig().console_mode) {
        // Send Python status to the visualiser
        data->modelCfg.isPython = isSWIG;
        // Init visualiser
        data->visualiser = std::make_unique<FLAMEGPU_Visualisation>(data->modelCfg);  // Window resolution
        data->visualiser->setRandomSeed(data->model.getSimulationConfig().random_seed);
        for (auto &agent : data->agents) {
            // If x and y aren't set, throw exception
            if (agent.second->core_tex_buffers.find(TexBufferConfig::Position_x) == agent.second->core_tex_buffers.end() &&
                agent.second->core_tex_buffers.find(TexBufferConfig::Position_y) == agent.second->core_tex_buffers.end() &&
                agent.second->core_tex_buffers.find(TexBufferConfig::Position_z) == agent.second->core_tex_buffers.end() &&
                agent.second->core_tex_buffers.find(TexBufferConfig::Position_xy) == agent.second->core_tex_buffers.end() &&
                agent.second->core_tex_buffers.find(TexBufferConfig::Position_xyz) == agent.second->core_tex_buffers.end()) {
                THROW exception::VisualisationException("Agent '%s' has not had x, y or z variables set, agent requires location to render, "
                    "in ModelVis::activate()\n",
                    agent.second->agentData->name.c_str());
            }
            agent.second->initBindings(data->visualiser);
        }
        data->env_registered = false;
        data->registerEnvProperties();
        data->visualiser->start();
    }
}
void ModelVis::deactivate() {
    if (data->visualiser && data->visualiser->isRunning()) {
        data->visualiser->stop();
        join();
        data->visualiser.reset();
    }
}

void ModelVis::join() {
    if (data->visualiser) {
        data->visualiser->join();
        data->visualiser.reset();
    }
}

bool ModelVis::isRunning() const {
    return data->visualiser ? data->visualiser->isRunning() : false;
}
void ModelVis::setWindowTitle(const std::string& title) {
    ModelConfig::setString(&data->modelCfg.windowTitle, title);
}

void ModelVis::setWindowDimensions(const unsigned int& width, const unsigned int& height) {
    data->modelCfg.windowDimensions[0] = width;
    data->modelCfg.windowDimensions[1] = height;
}

void ModelVis::setClearColor(const float& red, const float& green, const float& blue) {
    data->modelCfg.clearColor[0] = red;
    data->modelCfg.clearColor[1] = green;
    data->modelCfg.clearColor[2] = blue;
}

void ModelVis::setFPSVisible(const bool& showFPS) {
    data->modelCfg.fpsVisible = showFPS;
}

void ModelVis::setFPSColor(const float& red, const float& green, const float& blue) {
    data->modelCfg.fpsColor[0] = red;
    data->modelCfg.fpsColor[1] = green;
    data->modelCfg.fpsColor[2] = blue;
}

void ModelVis::setInitialCameraLocation(const float &x, const float &y, const float &z) {
    data->modelCfg.cameraLocation[0] = x;
    data->modelCfg.cameraLocation[1] = y;
    data->modelCfg.cameraLocation[2] = z;
}

void ModelVis::setInitialCameraTarget(const float &x, const float &y, const float &z) {
    data->modelCfg.cameraTarget[0] = x;
    data->modelCfg.cameraTarget[1] = y;
    data->modelCfg.cameraTarget[2] = z;
}

void ModelVis::setInitialCameraRoll(const float &roll) {
    data->modelCfg.cameraRoll = roll;
}

void ModelVis::setCameraSpeed(const float &speed, const float &shiftMultiplier) {
    data->modelCfg.cameraSpeed[0] = speed;
    data->modelCfg.cameraSpeed[1] = shiftMultiplier;
}

void ModelVis::setViewClips(const float &nearClip, const float &farClip) {
    data->modelCfg.nearFarClip[0] = nearClip;
    data->modelCfg.nearFarClip[1] = farClip;
}

void ModelVis::setOrthographic(const bool& isOrtho) {
    data->modelCfg.isOrtho = isOrtho;
}
void ModelVis::setOrthographicZoomModifier(const float& zoomMod) {
    data->modelCfg.orthoZoom = zoomMod;
}

void ModelVis::setStepVisible(const bool& showStep) {
    data->modelCfg.stepVisible = showStep;
}

void ModelVis::setSimulationSpeed(const unsigned int& _stepsPerSecond) {
    data->modelCfg.stepsPerSecond = _stepsPerSecond;
}

void ModelVis::setBeginPaused(const bool& beginPaused) {
    data->modelCfg.beginPaused = beginPaused;
}

StaticModelVis ModelVis::newStaticModel(const std::string &modelPath, const std::string &texturePath) {
    // Create ModelConfig::StaticModel
    auto m = std::make_shared<ModelConfig::StaticModel>();
    // set modelPath, texturePath
    m->path = modelPath;
    m->texture = texturePath;
    // add to ModelConfig.staticModels
    data->modelCfg.staticModels.push_back(m);
    // Create return type
    return StaticModelVis(data->modelCfg.staticModels.back());
}

LineVis ModelVis::newLineSketch(float r, float g, float b, float a) {
    auto m = std::make_shared<LineConfig>(LineConfig::Type::Lines);
    data->modelCfg.lines.push_back(m);
    return LineVis(m, r, g, b, a);
}

LineVis ModelVis::newPolylineSketch(float r, float g, float b, float a) {
    auto m = std::make_shared<LineConfig>(LineConfig::Type::Polyline);
    data->modelCfg.lines.push_back(m);
    return LineVis(m, r, g, b, a);
}
PanelVis ModelVis::newUIPanel(const std::string& panel_title) {
    if (data->modelCfg.panels.find(panel_title) != data->modelCfg.panels.end()) {
        THROW exception::InvalidOperation("Panel with title '%s' already exists.\n", panel_title.c_str());
    }
    auto m = std::make_shared<PanelConfig>(panel_title);
    data->modelCfg.panels.emplace(panel_title, m);
    return PanelVis(m, data->model.getModelDescription().environment);
}

}  // namespace visualiser
}  // namespace flamegpu
