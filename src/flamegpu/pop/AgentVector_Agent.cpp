#include "flamegpu/pop/AgentVector_Agent.h"

AgentVector_CAgent::AgentVector_CAgent(AgentVector* parent, const std::shared_ptr<const AgentData>& agent, const std::weak_ptr<AgentVector::AgentDataMap>& data, AgentVector::size_type pos)
    : index(pos)
    , _data(data)
    , _agent(agent)
, _parent(parent) { }
AgentVector_CAgent::~AgentVector_CAgent() {
}
id_t AgentVector_CAgent::getID() const {
    try {
        return getVariable< id_t>(ID_VARIABLE_NAME);
    } catch (ExpiredWeakPtr&) {
        throw;
    } catch (...) {
        // Rewrite all other exceptions
        THROW UnknownInternalError("Internal Error: Unable to read internal ID variable for agent '%s', in AgentVector::CAgent::getID()\n", _agent->name.c_str());
    }
}
AgentVector_Agent::AgentVector_Agent(AgentVector *parent, const std::shared_ptr<const AgentData>& agent, const std::weak_ptr<AgentVector::AgentDataMap>& data, AgentVector::size_type pos)
    : AgentVector_CAgent(parent, agent, data, pos) { }

void AgentVector_Agent::resetID() {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::resetID().\n");
    }
    const auto v_it = data->find(ID_VARIABLE_NAME);
    if (v_it == data->end()) {
        THROW InvalidOperation("Agent is missing internal ID variable, "
            "in AgentVector_Agent::resetID()\n");
    }
    auto& v_buff = v_it->second;
    // Don't bother checking type/elements
    _parent->_require(ID_VARIABLE_NAME);
    // do the replace
    static_cast<id_t*>(v_buff->getDataPtr())[index] = ID_NOT_SET;
    // Notify (_data was locked above)
    _parent->_changed(ID_VARIABLE_NAME, index);
}
