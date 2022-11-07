#include "flamegpu/model/ModelDescription.h"

#include "flamegpu/model/DependencyGraph.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/runtime/messaging/MessageBruteForce.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubEnvironmentDescription.h"

namespace flamegpu {

/**
* Constructors
*/
ModelDescription::ModelDescription(const std::string &model_name)
    : model(new ModelData(model_name)) { }

bool ModelDescription::operator==(const ModelDescription& rhs) const {
    return *this->model == *rhs.model;  // Compare content is functionally the same
}
bool ModelDescription::operator!=(const ModelDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Accessors
 */
AgentDescription ModelDescription::newAgent(const std::string &agent_name) {
    if (!hasAgent(agent_name)) {
        auto rtn = std::shared_ptr<AgentData>(new AgentData(model, agent_name));
        model->agents.emplace(agent_name, rtn);
        return AgentDescription(rtn);
    }
    THROW exception::InvalidAgentName("Agent with name '%s' already exists, "
        "in ModelDescription::newAgent().",
        agent_name.c_str());
}
AgentDescription ModelDescription::Agent(const std::string &agent_name) {
    auto rtn = model->agents.find(agent_name);
    if (rtn != model->agents.end())
        return AgentDescription(rtn->second);
    THROW exception::InvalidAgentName("Agent ('%s') was not found, "
        "in ModelDescription::Agent().",
        agent_name.c_str());
}

MessageBruteForce::Description& ModelDescription::newMessage(const std::string &message_name) {
    return newMessage<MessageBruteForce>(message_name);
}
MessageBruteForce::Description& ModelDescription::Message(const std::string &message_name) {
    return Message<MessageBruteForce>(message_name);
}

EnvironmentDescription& ModelDescription::Environment() {
    return *model->environment;
}

SubModelDescription ModelDescription::newSubModel(const std::string &submodel_name, const ModelDescription &submodel_description) {
    // Submodel is not self
    if (submodel_description.model == this->model) {
        THROW exception::InvalidSubModel("A model cannot be a submodel of itself, that would create infinite recursion, "
            "in ModelDescription::newSubModel().");
    }
    // Submodel is not already a submodel of this model
    for (auto &m : this->model->submodels) {
        if (m.second->submodel == submodel_description.model) {
            THROW exception::InvalidSubModel("Model '%s' is already a submodel of '%s', "
                "in ModelDescription::newSubModel().",
                submodel_name.c_str(), this->model->name.c_str());
        }
    }
    // Submodel is not already in the submodel hierarchy above us
    if (submodel_description.model->hasSubModelRecursive(this->model)) {
        THROW exception::InvalidSubModel("Models cannot exist in their own submodel hierarchy, that would create infinite recursion,"
            "in ModelDescription::newSubModel().");
    }
    // Submodel name is not in use
    if (!hasSubModel(submodel_name)) {
        auto rtn = std::shared_ptr<SubModelData>(new SubModelData(model, submodel_name, submodel_description.model));
        model->submodels.emplace(submodel_name, rtn);
        // This will actually generate the environment mapping (cant do it in constructor, due to shared_from_this)
        // Not the end of the world if it isn't init (we should be able to catch it down the line), but safer this way
        SubModelDescription rtn2(rtn);
        rtn2.SubEnvironment(false);
        return rtn2;
    }
    THROW exception::InvalidSubModelName("SubModel with name '%s' already exists, "
        "in ModelDescription::newSubModel().",
        submodel_name.c_str());
}
SubModelDescription ModelDescription::SubModel(const std::string &submodel_name) {
    auto rtn = model->submodels.find(submodel_name);
    if (rtn != model->submodels.end())
        return SubModelDescription(rtn->second);
    THROW exception::InvalidSubModelName("SubModel ('%s') was not found, "
        "in ModelDescription::SubModel().",
        submodel_name.c_str());
}

LayerDescription ModelDescription::newLayer(const std::string &name) {
    // Ensure name is unique
    if (!name.empty()) {
        for (auto it = model->layers.begin(); it != model->layers.end(); ++it) {
            if ((*it)->name == name) {
                THROW exception::InvalidFuncLayerIndx("Layer ('%s') already exists, "
                    "in ModelDescription::newLayer().",
                    name.c_str());
            }
        }
    }
    auto rtn = std::shared_ptr<LayerData>(new LayerData(model, name, static_cast<unsigned int>(model->layers.size())));
    model->layers.push_back(rtn);
    return LayerDescription(rtn);
}
LayerDescription ModelDescription::Layer(const flamegpu::size_type &layer_index) {
    if (model->layers.size() > layer_index) {
        auto it = model->layers.begin();
        for (auto i = 0u; i < layer_index; ++i)
            ++it;
        return LayerDescription(*it);
    }
    THROW exception::OutOfBoundsException("Layer %d is out of bounds, "
        "in ModelDescription::Layer().",
        layer_index);
}
LayerDescription ModelDescription::Layer(const std::string &name) {
    if (!name.empty()) {  // Can't search for no name, multiple layers might be nameless
        for (auto &layer : model->layers) {
            if (layer->name == name)
                return LayerDescription(layer);
        }
    }
    THROW exception::InvalidFuncLayerIndx("Layer '%s' was not found, "
        "in ModelDescription::Layer().",
        name.c_str());
}

void ModelDescription::addInitFunction(FLAMEGPU_INIT_FUNCTION_POINTER func_p) {
    if (std::find(model->initFunctions.begin(), model->initFunctions.end(), func_p) != model->initFunctions.end()) {
        THROW exception::InvalidHostFunc("Attempted to add same init function twice,"
            "in ModelDescription::addInitFunction()");
    }
    model->initFunctions.push_back(func_p);
}
void ModelDescription::addStepFunction(FLAMEGPU_STEP_FUNCTION_POINTER func_p) {
    if (std::find(model->stepFunctions.begin(), model->stepFunctions.end(), func_p) != model->stepFunctions.end()) {
        THROW exception::InvalidHostFunc("Attempted to add same step function twice,"
            "in ModelDescription::addStepFunction()");
    }
    model->stepFunctions.push_back(func_p);
}
void ModelDescription::addExitFunction(FLAMEGPU_EXIT_FUNCTION_POINTER func_p) {
    if (std::find(model->exitFunctions.begin(), model->exitFunctions.end(), func_p) != model->exitFunctions.end()) {
        THROW exception::InvalidHostFunc("Attempted to add same exit function twice,"
            "in ModelDescription::addExitFunction()");
    }
    model->exitFunctions.push_back(func_p);
}

void ModelDescription::addExitCondition(FLAMEGPU_EXIT_CONDITION_POINTER func_p) {
    if (std::find(model->exitConditions.begin(), model->exitConditions.end(), func_p) != model->exitConditions.end()) {
        THROW exception::InvalidHostFunc("Attempted to add same exit condition twice,"
            "in ModelDescription::addExitCondition()");
    }
    model->exitConditions.push_back(func_p);
}

void ModelDescription::addExecutionRoot(DependencyNode& root) {
    model->dependencyGraph->addRoot(root);
}

void ModelDescription::generateLayers() {
    model->dependencyGraph->generateLayers();
}

/**
 * Accessors
 */
const DependencyGraph& ModelDescription::getDependencyGraph() const {
    return *(model->dependencyGraph);
}

/**
* Const Accessors
*/
std::string ModelDescription::getName() const {
    return model->name;
}

CAgentDescription ModelDescription::getAgent(const std::string& agent_name) const {
    const auto rtn = model->agents.find(agent_name);
    if (rtn != model->agents.end())
        return CAgentDescription(rtn->second);
    THROW exception::InvalidAgentName("Agent ('%s') was not found, "
        "in ModelDescription::getAgent().",
        agent_name.c_str());
}
const MessageBruteForce::Description& ModelDescription::getMessage(const std::string &message_name) const {
    return getMessage<MessageBruteForce>(message_name);
}
CSubModelDescription ModelDescription::getSubModel(const std::string &submodel_name) const {
    const auto rtn = model->submodels.find(submodel_name);
    if (rtn != model->submodels.end())
        return CSubModelDescription(rtn->second);
    THROW exception::InvalidSubModelName("SubModel ('%s') was not found, "
        "in ModelDescription::getSubModel().",
        submodel_name.c_str());
}
const EnvironmentDescription& ModelDescription::getEnvironment() const {
    return *model->environment;
}
CLayerDescription ModelDescription::getLayer(const std::string &name) const {
    if (!name.empty()) {  // Can't search for no name, multiple layers might be nameless
        for (auto it = model->layers.begin(); it != model->layers.end(); ++it) {
            if ((*it)->name == name)
                return CLayerDescription(*it);
        }
    }
    THROW exception::InvalidFuncLayerIndx("Layer ('%s') was not found, "
        "in ModelDescription::getAgent().",
        name.c_str());
}
CLayerDescription ModelDescription::getLayer(const flamegpu::size_type &layer_index) const {
    if (model->layers.size() > layer_index) {
        auto it = model->layers.begin();
        for (auto i = 0u; i < layer_index; ++i)
            ++it;
        return CLayerDescription(*it);
    }
    THROW exception::OutOfBoundsException("Layer %d is out of bounds, "
        "in ModelDescription::Layer().",
        layer_index);
}
bool ModelDescription::hasAgent(const std::string &agent_name) const {
    return model->agents.find(agent_name) != model->agents.end();
}
bool ModelDescription::hasMessage(const std::string &message_name) const {
    return model->messages.find(message_name) != model->messages.end();
}
bool ModelDescription::hasSubModel(const std::string &submodel_name) const {
    return model->submodels.find(submodel_name) != model->submodels.end();
}

bool ModelDescription::hasLayer(const std::string &name) const {
    if (!name.empty()) {  // Can't search for no name, multiple layers might be nameless
        for (auto it = model->layers.begin(); it != model->layers.end(); ++it) {
            if ((*it)->name == name)
                return true;
        }
    }
    return false;
}
bool ModelDescription::hasLayer(const flamegpu::size_type &layer_index) const {
    return layer_index < model->layers.size();
}

void ModelDescription::generateDependencyGraphDOTDiagram(std::string outputFileName) const {
    model->dependencyGraph->generateDOTDiagram(outputFileName);
}
std::string ModelDescription::getConstructedLayersString() const {
    return model->dependencyGraph->getConstructedLayersString();
}

flamegpu::size_type ModelDescription::getAgentsCount() const {
    // This down-cast is safe
    return static_cast<flamegpu::size_type>(model->agents.size());
}
flamegpu::size_type ModelDescription::getMessagesCount() const {
    // This down-cast is safe
    return static_cast<flamegpu::size_type>(model->messages.size());
}
flamegpu::size_type ModelDescription::getLayersCount() const {
    // This down-cast is safe
    return static_cast<flamegpu::size_type>(model->layers.size());
}

}  // namespace flamegpu
