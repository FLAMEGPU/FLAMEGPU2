#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/runtime/messaging/BruteForce.h"


#include <iostream>

const char *ModelData::DEFAULT_STATE = "default";

/**
 * Constructors
 */
ModelData::ModelData(const std::string &model_name)
    : environment(new EnvironmentDescription())
    , name(model_name) { }

ModelData::~ModelData() { }

AgentData::AgentData(ModelData *const model, const std::string &agent_name)
    : initial_state(ModelData::DEFAULT_STATE)
    , agent_outputs(0)
    , description(new AgentDescription(model, this))
    , name(agent_name)
    , keepDefaultState(false) {
    states.insert(ModelData::DEFAULT_STATE);
}


AgentFunctionData::AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string &function_name, AgentFunctionWrapper *agent_function, const std::string &in_type, const std::string &out_type)
    : func(agent_function)
    , rtc_source("")
    , initial_state(_parent->initial_state)
    , end_state(_parent->initial_state)
    , message_output_optional(false)
    , has_agent_death(false)
    , condition(nullptr)
    , parent(_parent)
    , description(new AgentFunctionDescription(_parent->description->model, this))
    , name(function_name)
    , msg_in_type(in_type)
    , msg_out_type(out_type) { }

AgentFunctionData::AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string& function_name, const std::string &rtc_function_src, const std::string &in_type, const std::string &out_type)
    : func(0)
    , rtc_source(rtc_function_src)
    , initial_state(_parent->initial_state)
    , end_state(_parent->initial_state)
    , message_output_optional(false)
    , has_agent_death(false)
    , condition(nullptr)
    , parent(_parent)
    , description(new AgentFunctionDescription(_parent->description->model, this))
    , name(function_name)
    , msg_in_type(in_type)
    , msg_out_type(out_type) { }

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
        messages.emplace(m.first, std::shared_ptr<MsgBruteForce::Data>(m.second->clone(this)));  // Need to convert this to shared_ptr, how to force shared copy construct?
    }
    // Copy all agents first
    for (const auto a : other.agents) {
        auto b = std::shared_ptr<AgentData>(new AgentData(this, *a.second));
        agents.emplace(a.first, b);
    }
    // Copy agent functions per agent, after all agents have been implemented.
    for (const auto a : other.agents) {
        auto b = agents.find(a.first)->second;
        // Manually copy construct maps of shared ptr
        for (const auto f : a.second->functions) {
            b->functions.emplace(f.first, std::shared_ptr<AgentFunctionData>(new AgentFunctionData(this, b, *f.second)));
        }
    }
    for (const auto m : other.layers) {
        layers.push_back(std::shared_ptr<LayerData>(new LayerData(this, *m)));
    }
}

std::shared_ptr<const AgentData> AgentData::clone() const {
    std::shared_ptr<AgentData> b = std::shared_ptr<AgentData>(new AgentData(nullptr, *this));
    // Manually copy construct maps of shared ptr
    for (const auto f : functions) {
        b->functions.emplace(f.first, std::shared_ptr<AgentFunctionData>(new AgentFunctionData(nullptr, b, *f.second)));
    }
    return b;
}
AgentData::AgentData(ModelData *const model, const AgentData &other)
    : variables(other.variables)
    , states(other.states)
    , initial_state(other.initial_state)
    , agent_outputs(other.agent_outputs)
    , description(model ? new AgentDescription(model, this) : nullptr)
    , name(other.name)
    , keepDefaultState(other.keepDefaultState) { }
AgentFunctionData::AgentFunctionData(ModelData *const model, std::shared_ptr<AgentData> _parent, const AgentFunctionData &other)
    : func(other.func)
    , initial_state(other.initial_state)
    , end_state(other.end_state)
    , message_output_optional(other.message_output_optional)
    , agent_output_state(other.agent_output_state)
    , has_agent_death(other.has_agent_death)
    , condition(other.condition)
    , parent(_parent)
    , description(model ? new AgentFunctionDescription(model, this) : nullptr)
    , name(other.name)
    , msg_in_type(other.msg_in_type)
    , msg_out_type(other.msg_out_type) {
    // Manually perform lookup copies
    if (model) {
        std::cout << "Demangled name: " << demangle(std::type_index(typeid(MsgNone)).name()) << "\n";
        if (auto a = other.message_input.lock()) {
            auto _m = model->messages.find(a->name);
            if (_m != model->messages.end()) {
                message_input = _m->second;
            }
        } else if (other.msg_in_type != demangle(std::type_index(typeid(MsgNone)).name())) {
            THROW InvalidMessageType("Function '%s' is missing bound input message of type '%s', type provided was '%s'.", other.name.c_str(), other.msg_in_type.c_str(), demangle(std::type_index(typeid(MsgNone)).name()).c_str());
        }
        if (auto a = other.message_output.lock()) {
            auto _m = model->messages.find(a->name);
            if (_m != model->messages.end()) {
                message_output = _m->second;
            }
        } else if (other.msg_out_type != demangle(std::type_index(typeid(MsgNone)).name())) {
            THROW InvalidMessageType("Function '%s' is missing bound output message of type '%s'.", other.name.c_str(), other.msg_out_type.c_str(), demangle(std::type_index(typeid(MsgNone)).name()).c_str());
        }
        if (auto a = other.agent_output.lock()) {
            auto _a = model->agents.find(a->name);
            if (_a != model->agents.end()) {
                agent_output = _a->second;
            }
        }
    }
}
LayerData::LayerData(ModelData *const model, const LayerData &other)
    : host_functions(other.host_functions)
    , description(model ? new LayerDescription(model, this) : nullptr)
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
    next_agent_fn: {}
    }
}


bool ModelData::operator==(const ModelData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name
        && agents.size() == rhs.agents.size()
        && messages.size() == rhs.messages.size()
        && layers.size() == rhs.layers.size()
        && initFunctions.size() == rhs.initFunctions.size()
        && stepFunctions.size() == rhs.stepFunctions.size()
        && exitFunctions.size() == rhs.exitFunctions.size()
        && exitConditions.size() == rhs.exitConditions.size()
        && *environment == *rhs.environment) {
            {  // Compare agents (map)
                for (auto &v : agents) {
                    auto _v = rhs.agents.find(v.first);
                    if (_v == rhs.agents.end())
                        return false;
                    if (*v.second != *_v->second)
                        return false;
                }
            }
            {  // Compare messages (map)
                for (auto &v : messages) {
                    auto _v = rhs.messages.find(v.first);
                    if (_v == rhs.messages.end())
                        return false;
                    if (*v.second != *_v->second)
                        return false;
                }
            }
            {  // Compare layers (ordered list)
                auto it1 = layers.begin();
                auto it2 = rhs.layers.begin();
                while (it1 != layers.end() && it2 != rhs.layers.end()) {
                    if (*it1 != *it2)
                        return false;
                    ++it1;
                    ++it2;
                }
            }
            {  // Init fns (set)
                if (initFunctions != rhs.initFunctions)
                    return false;
            }
            {  // Step fns (set)
                if (stepFunctions != rhs.stepFunctions)
                    return false;
            }
            {  // Exit fns (set)
                if (exitFunctions != rhs.exitFunctions)
                    return false;
            }
            {  // Exit cdns (set)
                if (exitConditions != rhs.exitConditions)
                    return false;
            }
            return true;
    }
    return false;
}
bool ModelData::operator!=(const ModelData& rhs) const {
    return !operator==(rhs);
}
bool AgentData::operator==(const AgentData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name
        && initial_state == rhs.initial_state
        && agent_outputs == rhs.agent_outputs
        && keepDefaultState == rhs.keepDefaultState
        && functions.size() == rhs.functions.size()
        && variables.size() == rhs.variables.size()
        && states.size() == rhs.states.size()) {
        {  // Compare functions
                for (auto &v : functions) {
                    auto _v = rhs.functions.find(v.first);
                    if (_v == rhs.functions.end())
                        return false;
                    if (v.second->operator==(*_v->second))
                        return false;
                }
        }
        {  // Compare variables
            for (auto &v : variables) {
                auto _v = rhs.variables.find(v.first);
                if (_v == rhs.variables.end())
                    return false;
                if (v.second.type_size != _v->second.type_size
                    || v.second.type != _v->second.type
                    || v.second.elements != _v->second.elements)
                    return false;
            }
        }
        {  // Compare state lists
            if (states != rhs.states)
                return false;
        }
        return true;
    }
    return false;
}
bool AgentData::operator!=(const AgentData& rhs) const {
    return !operator==(rhs);
}
bool AgentFunctionData::operator==(const AgentFunctionData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if ((name == rhs.name)
        && (func == rhs.func)
        && (initial_state == rhs.initial_state)
        && (end_state == rhs.end_state)
        && (message_output_optional == rhs.message_output_optional)
        && (has_agent_death == rhs.has_agent_death)) {
        // Test weak pointers
            {  // message_input
                auto a = message_input.lock();
                auto b = rhs.message_input.lock();
                if (a && b) {
                    if (*a != *b)
                        return false;
                } else if ((a && !b) || (!a && !b)) {
                    return false;
                }
            }
            {  // message_output
                auto a = message_output.lock();
                auto b = rhs.message_output.lock();
                if (a && b) {
                    if (*a != *b)
                        return false;
                } else if ((a && !b) || (!a && !b)) {
                    return false;
                }
            }
            {  // agent_output
                auto a = agent_output.lock();
                auto b = rhs.agent_output.lock();
                if (a && b) {  // Comparing agents here is unsafe, as agent might link back to this same function, so do a reduced comparison
                    if (a->name != b->name
                        || a->initial_state != b->initial_state
                        || a->agent_outputs != b->agent_outputs
                        || a->keepDefaultState != b->keepDefaultState
                        || a->states.size() != b->states.size()
                        || a->functions.size() != b->functions.size()
                        || a->variables.size() != b->variables.size()
                        || a->states != b->states)
                        return false;
                    {  // Compare agent variables
                        for (auto &v : a->variables) {
                            auto _v = b->variables.find(v.first);
                            if (_v == b->variables.end())
                                return false;
                            if (v.second.type_size != _v->second.type_size
                                || v.second.type != _v->second.type
                                || v.second.elements != _v->second.elements)
                                return false;
                        }
                    }
                    {  // Compare agent functions (compare name only)
                        for (auto &v : a->functions) {
                            auto _v = b->functions.find(v.first);
                            if (_v == b->functions.end())
                                return false;
                        }
                    }
                } else if ((a && !b) || (!a && !b)) {
                    return false;
                }
            }
            return true;
    }
    return false;
}
bool AgentFunctionData::operator!=(const AgentFunctionData& rhs) const {
    return !operator==(rhs);
}
bool LayerData::operator==(const LayerData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name
        && index == rhs.index
        && agent_functions.size() == rhs.agent_functions.size()
        && host_functions.size() == rhs.host_functions.size()
        && agent_functions == rhs.agent_functions
        && host_functions == rhs.host_functions) {
            return true;
    }
    return false;
}
bool LayerData::operator!=(const LayerData& rhs) const {
    return !operator==(rhs);
}

bool AgentData::isOutputOnDevice() const {
    return agent_outputs > 0;
}
