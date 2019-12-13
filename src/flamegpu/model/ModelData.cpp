#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/LayerDescription.h"

const std::string ModelData::DEFAULT_STATE = "default";

/**
 * Constructors
 */
ModelData::ModelData(const std::string &model_name)
    : environment(new EnvironmentDescription())
    , name(model_name) { }

AgentData::AgentData(ModelData *const model, const std::string &agent_name)
    : agent_outputs(0)
    , description(new AgentDescription(model, this))
    , name(agent_name)
    , keepDefaultState(false) {
    states.insert(ModelData::DEFAULT_STATE);
}

MessageData::MessageData(ModelData *const model, const std::string &message_name)
    : description(new MessageDescription(model, this))
    , name(message_name) { }

AgentFunctionData::AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string &function_name, AgentFunctionWrapper *agent_function)
    : func(agent_function)
    , initial_state(ModelData::DEFAULT_STATE)
    , end_state(ModelData::DEFAULT_STATE)
    , message_output_optional(false)
    , has_agent_death(false)
    , parent(_parent)
    , description(new AgentFunctionDescription(_parent->description->model, this))
    , name(function_name) { }

LayerData::LayerData(ModelData *const model, const std::string &layer_name, const ModelData::size_type &layer_index)
    : description(new LayerDescription(model, this))
    , name(layer_name)
    , index(layer_index) { }


/**
 * Copy Constructors
 */
std::shared_ptr<ModelData> ModelData::clone() const {
    // Arkwardly cant use shared from this inside constructor, so use raw pts instead
    return std::shared_ptr<ModelData>(new ModelData(*this));
}
ModelData::ModelData(const ModelData &other)
    : initFunctions(other.initFunctions)
    , stepFunctions(other.stepFunctions)
    , exitFunctions(other.exitFunctions)
    , exitConditions(other.exitConditions)
    , environment(new EnvironmentDescription(*other.environment))
    , name(other.name) {
    // Manually copy construct maps of shared ptr
    for (const auto m : other.messages) {
        messages.emplace(m.first, std::shared_ptr<MessageData>(new MessageData(this, *m.second)));//Need to convert this to shared_ptr, how to force shared copy construct?
    }
    for (const auto a : other.agents) {
        auto b = std::shared_ptr<AgentData>(new AgentData(this, *a.second));
        agents.emplace(a.first, b);
        // Manually copy construct maps of shared ptr
        for (const auto f : a.second->functions) {
            b->functions.emplace(f.first, std::shared_ptr<AgentFunctionData>(new AgentFunctionData(this, b, *f.second)));
        }
    }
    for (const auto m : other.layers) {
        layers.push_back(std::shared_ptr<LayerData>(new LayerData(this, *m)));
    }
}

AgentData::AgentData(ModelData *const model, const AgentData &other)
    : variables(other.variables)
    , states(other.states)
    , initial_state(other.initial_state)
    , agent_outputs(other.agent_outputs)
    , description(new AgentDescription(model, this)) 
    , name(other.name) 
    , keepDefaultState(other.keepDefaultState) {
}
MessageData::MessageData(ModelData *const model, const MessageData &other)
    : variables(other.variables)
    , description(new MessageDescription(model, this))
    , name(other.name) {

}
AgentFunctionData::AgentFunctionData(ModelData *const model, std::shared_ptr<AgentData> _parent, const AgentFunctionData &other)
    : func (other.func)
    , initial_state(other.initial_state)
    , end_state(other.end_state)
    , message_output_optional(other.message_output_optional)
    , has_agent_death(other.has_agent_death)
    , parent(_parent)
    , description(new AgentFunctionDescription(model, this))
    , name(other.name) {
    // Manually perform lookup copies
    if (auto a = other.message_input.lock()) {
        auto _m = model->messages.find(a->name);
        if (_m != model->messages.end()) {
            message_input = _m->second;
        }
    }
    if (auto a = other.message_output.lock()) {
        auto _m = model->messages.find(a->name);
        if (_m != model->messages.end()) {
            message_output = _m->second;
        }
    }
    if (auto a = other.agent_output.lock()) {
        auto _a = model->agents.find(a->name);
        if (_a != model->agents.end()) {
            agent_output = _a->second;
        }
    }
}
LayerData::LayerData(ModelData *const model, const LayerData &other)
    : host_functions(other.host_functions)
    , description(new LayerDescription(model, this))
    , name(other.name)
    , index(other.index) {
    // Manually perform lookup copies
    for (auto _f : other.agent_functions) {
        for (auto a : model->agents) {
            for (auto f : a.second->functions) {
                if (f.second->func == _f->func) {
                    if (f.second->name == _f->name) {
                        agent_functions.emplace(f.second);
                        goto next_agent_fn;
                    }
                }
            }
        }
    next_agent_fn:;
    }
}

bool AgentData::operator==(const AgentData& rhs) const
{
    if (name == rhs.name
        && initial_state == rhs.initial_state
        && states.size() == rhs.states.size()
        && variables.size() == rhs.variables.size()) {
        {  // Compare state lists
            auto it1 = states.begin();
            auto it2 = rhs.states.begin();
            while (it1 != states.end() && it2 != rhs.states.end()) {
                if (*it1 != *it2)
                    return false;
                ++it1;
                ++it2;
            }
        }
        {  // Compare variables
            for(auto &v:variables) {
                auto _v = rhs.variables.find(v.first);
                if (_v == rhs.variables.end())
                    return false;
                if (v.second.type_size != _v->second.type_size
                    || v.second.type != _v->second.type
                    || v.second.elements != _v->second.elements)
                    return false;
            }
        }
        return true;
    }
    return false;
}
bool AgentData::operator!=(const AgentData& rhs) const
{
    return !operator==(rhs);
}
bool AgentFunctionData::operator==(const AgentFunctionData& rhs) const
{
    return (name == rhs.name)
        && (func == rhs.func); // This is the only comparison that matters
}
bool AgentFunctionData::operator!=(const AgentFunctionData& rhs) const
{
    return !operator==(rhs);
}

bool AgentData::isOutputOnDevice() const {
    return agent_outputs > 0;
}