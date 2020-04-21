#include "flamegpu/model/ModelDescription.h"

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/model/SubModelData.h"

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
AgentDescription& ModelDescription::newAgent(const std::string &agent_name) {
    if (!hasAgent(agent_name)) {
        auto rtn = std::shared_ptr<AgentData>(new AgentData(model, agent_name));
        model->agents.emplace(agent_name, rtn);
        return *rtn->description;
    }
    THROW InvalidAgentName("Agent with name '%s' already exists, "
        "in ModelDescription::newAgent().",
        agent_name.c_str());
}
AgentDescription& ModelDescription::Agent(const std::string &agent_name) {
    auto rtn = model->agents.find(agent_name);
    if (rtn != model->agents.end())
        return *rtn->second->description;
    THROW InvalidAgentName("Agent ('%s') was not found, "
        "in ModelDescription::Agent().",
        agent_name.c_str());
}

MsgBruteForce::Description& ModelDescription::newMessage(const std::string &message_name) {
    return newMessage<MsgBruteForce>(message_name);
}
MsgBruteForce::Description& ModelDescription::Message(const std::string &message_name) {
    return Message<MsgBruteForce>(message_name);
}

EnvironmentDescription& ModelDescription::Environment() {
    return *model->environment;
}

SubModelDescription& ModelDescription::newSubModel(const std::string &submodel_name, const ModelDescription &submodel_description) {
    if (!hasSubModel(submodel_name)) {
        if (submodel_description.model->exitConditions.empty()) {
            THROW InvalidSubModel("Model '%s' does not contain any exit conditions, SubModels must exit of their own accord, "
                "in ModelDescription::newSubModel().",
                submodel_name.c_str());
        }
        auto rtn = std::shared_ptr<SubModelData>(new SubModelData(model, submodel_description.model->clone()));
        model->submodels.emplace(submodel_name, rtn);
        return *rtn->description;
    }
    THROW InvalidSubModelName("SubModel with name '%s' already exists, "
        "in ModelDescription::newSubModel().",
        submodel_name.c_str());
}
SubModelDescription &ModelDescription::SubModel(const std::string &submodel_name) {
    auto rtn = model->submodels.find(submodel_name);
    if (rtn != model->submodels.end())
        return *rtn->second->description;
    THROW InvalidSubModelName("SubModel ('%s') was not found, "
        "in ModelDescription::SubModel().",
        submodel_name.c_str());
}

LayerDescription& ModelDescription::newLayer(const std::string &name) {
    // Ensure name is unique
    if (!name.empty()) {
        for (auto it = model->layers.begin(); it != model->layers.end(); ++it) {
            if ((*it)->name == name) {
                THROW InvalidFuncLayerIndx("Layer ('%s') already exists, "
                    "in ModelDescription::newLayer().",
                    name.c_str());
            }
        }
    }
    auto rtn = std::shared_ptr<LayerData>(new LayerData(model, name, static_cast<unsigned int>(model->layers.size())));
    model->layers.push_back(rtn);
    return *rtn->description;
}
LayerDescription& ModelDescription::Layer(const ModelData::size_type &layer_index) {
    if (model->layers.size() > layer_index) {
        auto it = model->layers.begin();
        for (auto i = 0u; i < layer_index; ++i)
            ++it;
        return *(*it)->description;
    }
    THROW OutOfBoundsException("Layer %d is out of bounds, "
        "in ModelDescription::Layer().",
        layer_index);
}
LayerDescription& ModelDescription::Layer(const std::string &name) {
    if (!name.empty()) {  // Can't search for no name, multiple layers might be nameless
        for (auto &layer : model->layers) {
            if (layer->name == name)
                return *layer->description;
        }
    }
    THROW InvalidFuncLayerIndx("Layer '%s' was not found, "
        "in ModelDescription::Layer().",
        name.c_str());
}

void ModelDescription::addInitFunction(FLAMEGPU_INIT_FUNCTION_POINTER func_p) {
    if (!model->initFunctions.insert(func_p).second) {
        THROW InvalidHostFunc("Attempted to add same init function twice,"
            "in ModelDescription::addInitFunction()");
    }
}
void ModelDescription::addStepFunction(FLAMEGPU_STEP_FUNCTION_POINTER func_p) {
    if (!model->stepFunctions.insert(func_p).second) {
        THROW InvalidHostFunc("Attempted to add same step function twice,"
            "in ModelDescription::addStepFunction()");
    }
}
void ModelDescription::addExitFunction(FLAMEGPU_EXIT_FUNCTION_POINTER func_p) {
    if (!model->exitFunctions.insert(func_p).second) {
        THROW InvalidHostFunc("Attempted to add same exit function twice,"
            "in ModelDescription::addExitFunction()");
    }
}
void ModelDescription::addExitCondition(FLAMEGPU_EXIT_CONDITION_POINTER func_p) {
    if (!model->exitConditions.insert(func_p).second) {
        THROW InvalidHostFunc("Attempted to add same exit condition twice,"
            "in ModelDescription::addExitCondition()");
    }
}

/**
* Const Accessors
*/
std::string ModelDescription::getName() const {
    return model->name;
}

const AgentDescription& ModelDescription::getAgent(const std::string &agent_name) const {
    const auto rtn = model->agents.find(agent_name);
    if (rtn != model->agents.end())
        return *rtn->second->description;
    THROW InvalidAgentName("Agent ('%s') was not found, "
        "in ModelDescription::getAgent().",
        agent_name.c_str());
}
const MsgBruteForce::Description& ModelDescription::getMessage(const std::string &message_name) const {
    return getMessage<MsgBruteForce>(message_name);
}
const SubModelDescription &ModelDescription::getSubModel(const std::string &submodel_name) const {
    const auto rtn = model->submodels.find(submodel_name);
    if (rtn != model->submodels.end())
        return *rtn->second->description;
    THROW InvalidSubModelName("SubModel ('%s') was not found, "
        "in ModelDescription::getSubModel().",
        submodel_name.c_str());
}
const EnvironmentDescription& ModelDescription::getEnvironment() const {
    return *model->environment;
}
const LayerDescription& ModelDescription::getLayer(const std::string &name) const {
    if (!name.empty()) {  // Can't search for no name, multiple layers might be nameless
        for (auto it = model->layers.begin(); it != model->layers.end(); ++it) {
            if ((*it)->name == name)
                return *(*it)->description;
        }
    }
    THROW InvalidFuncLayerIndx("Layer ('%s') was not found, "
        "in ModelDescription::getAgent().",
        name.c_str());
}
const LayerDescription& ModelDescription::getLayer(const ModelData::size_type &layer_index) const {
    if (model->layers.size() > layer_index) {
        auto it = model->layers.begin();
        for (auto i = 0u; i < layer_index; ++i)
            ++it;
        return *(*it)->description;
    }
    THROW OutOfBoundsException("Layer %d is out of bounds, "
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
bool ModelDescription::hasLayer(const ModelData::size_type &layer_index) const {
    return layer_index < model->layers.size();
}

ModelData::size_type ModelDescription::getAgentsCount() const {
    // This down-cast is safe
    return static_cast<ModelData::size_type>(model->agents.size());
}
ModelData::size_type ModelDescription::getMessagesCount() const {
    // This down-cast is safe
    return static_cast<ModelData::size_type>(model->messages.size());
}
ModelData::size_type ModelDescription::getLayersCount() const {
    // This down-cast is safe
    return static_cast<ModelData::size_type>(model->layers.size());
}
