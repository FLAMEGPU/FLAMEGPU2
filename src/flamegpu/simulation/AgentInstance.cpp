#include "flamegpu/runtime/agent/AgentInstance.h"

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/simulation/AgentVector.h"

namespace flamegpu {

AgentInstance::AgentInstance(const CAgentDescription& agent_desc)
    : _agent(agent_desc.agent->clone()) {
    // Fill data map with default values
    for (const auto& v : _agent->variables) {
        _data.emplace(v.first, detail::Any(v.second.default_value, v.second.type_size * v.second.elements, v.second.type, v.second.elements));
    }
}

AgentInstance::AgentInstance(const AgentInstance& other)
    : _data(other._data)  // copy-assignment
    , _agent(other._agent->clone()) { }

AgentInstance::AgentInstance(AgentInstance&& other) noexcept {
    std::swap(_agent, other._agent);
    std::swap(_data, other._data);
}
AgentInstance::AgentInstance(const AgentVector::CAgent& other)
    : _agent(other._agent->clone()) {
    const auto other_data = other._data.lock();
    if (!other_data) {
        THROW exception::ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentInstance::AgentInstance().\n");
    }
    // Copy data items manually because format change
    for (const auto& v : _agent->variables) {
        const auto &it = other_data->at(v.first);
        const auto variable_size = v.second.elements * v.second.type_size;
        _data.emplace(v.first, detail::Any(static_cast<const char*>(it->getReadOnlyDataPtr()) + other.index * variable_size,
            variable_size, it->getType(), it->getElements()));
    }
}


AgentInstance& AgentInstance::operator=(const AgentInstance& other) {
    // Self assignment
    if (this == &other)
        return *this;
    // if (*_agent != *other._agent) {
    //    throw std::exception();  // AgentInstance are for different AgentDescriptions
    // }
    _agent = other._agent->clone();
    _data = other._data;  // copy-assignment
    return *this;
}
AgentInstance& AgentInstance::operator=(AgentInstance&& other) noexcept {
    std::swap(_agent, other._agent);
    std::swap(_data, other._data);
    return *this;
}
AgentInstance& AgentInstance::operator=(const AgentVector::CAgent& other) {
    _agent = other._agent->clone();
    _data.clear();
    const auto other_data = other._data.lock();
    if (!other_data) {
        THROW exception::ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
        "in AgentInstance::operator=().\n");
    }
    // Copy data items manually because format change
    for (const auto& v : _agent->variables) {
        const auto& it = other_data->at(v.first);
        const auto variable_size = v.second.elements * v.second.type_size;
        _data.emplace(v.first, detail::Any(static_cast<const char*>(it->getReadOnlyDataPtr()) + other.index * variable_size,
            variable_size, it->getType(), it->getElements()));
    }
    return *this;
}

}  // namespace flamegpu
