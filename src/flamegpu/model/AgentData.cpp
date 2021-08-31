#include "flamegpu/model/AgentData.h"

#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

namespace flamegpu {

AgentData::AgentData(std::shared_ptr<const ModelData> model, const std::string &agent_name)
    : initial_state(ModelData::DEFAULT_STATE)
    , agent_outputs(0)
    , description(new AgentDescription(model, this))
    , name(agent_name)
    , keepDefaultState(false) {
    states.insert(ModelData::DEFAULT_STATE);
    // All agents have an internal _id variable
    variables.emplace(ID_VARIABLE_NAME, Variable(std::array<id_t, 1>{ ID_NOT_SET }));
}

std::shared_ptr<const AgentData> AgentData::clone() const {
    std::shared_ptr<AgentData> b = std::shared_ptr<AgentData>(new AgentData(nullptr, *this));
    // Manually copy construct maps of shared ptr
    for (const auto &f : functions) {
        // Passing model is risky here, as the weak_ptr for agent output will point here
        b->functions.emplace(f.first, std::shared_ptr<AgentFunctionData>(new AgentFunctionData(description->model.lock(), b, *f.second)));
    }
    return b;
}
AgentData::AgentData(std::shared_ptr<const ModelData> model, const AgentData &other)
    : variables(other.variables)
    , states(other.states)
    , initial_state(other.initial_state)
    , agent_outputs(other.agent_outputs)
    , description(model ? new AgentDescription(model, this) : nullptr)
    , name(other.name)
    , keepDefaultState(other.keepDefaultState) { }

bool AgentData::operator==(const AgentData &rhs) const {
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
                if (*v.second != *_v->second)
                    return false;
            }
        }
        {  // Compare variables
            for (auto &v : variables) {
                auto _v = rhs.variables.find(v.first);
                if (_v == rhs.variables.end())
                    return false;
                if (v.second.type_size != _v->second.type_size || v.second.type != _v->second.type || v.second.elements != _v->second.elements)
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

bool AgentData::operator!=(const AgentData &rhs) const {
    return !operator==(rhs);
}

bool AgentData::isOutputOnDevice() const {
    return agent_outputs > 0;
}

}  // namespace flamegpu
