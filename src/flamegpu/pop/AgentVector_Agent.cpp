#include "flamegpu/pop/AgentVector_Agent.h"

AgentVector_CAgent::AgentVector_CAgent(AgentVector* parent, const std::shared_ptr<const AgentData>& agent, const std::weak_ptr<AgentVector::AgentDataMap>& data, AgentVector::size_type pos)
    : index(pos)
    , _data(data)
    , _agent(agent)
, _parent(parent) { }
AgentVector_CAgent::~AgentVector_CAgent() {
}
AgentVector_Agent::AgentVector_Agent(AgentVector *parent, const std::shared_ptr<const AgentData>& agent, const std::weak_ptr<AgentVector::AgentDataMap>& data, AgentVector::size_type pos)
    : AgentVector_CAgent(parent, agent, data, pos) { }
