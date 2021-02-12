#ifndef INCLUDE_FLAMEGPU_POP_AGENTVECTOR_AGENT_H_
#define INCLUDE_FLAMEGPU_POP_AGENTVECTOR_AGENT_H_

/**
 * THIS CLASS SHOULD NOT BE INCLUDED DIRECTLY
 * Include flamegpu/pop/AgentVector.h instead
 * Use AgentVector::CAgent instead of AgentVector_CAgent
 * Use AgentVector::Agent instead of AgentVector_Agent
 */

#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "flamegpu/pop/AgentVector.h"

class AgentInstance;

/**
 * const view into AgentVector
 */
class AgentVector_CAgent {
    friend AgentVector::CAgent AgentVector::at(AgentVector::size_type) const;
    friend AgentVector::CAgent AgentVector::const_iterator::operator*() const;
    friend AgentVector::CAgent AgentVector::const_reverse_iterator::operator*() const;
    friend AgentVector::iterator AgentVector::insert(AgentVector::size_type pos, AgentVector::size_type count, const AgentVector::Agent&);
    friend class AgentInstance;
    // friend AgentInstance::AgentInstance(const AgentVector::CAgent&);
    // friend AgentInstance& AgentInstance::operator=(const AgentVector::CAgent&);

 public:
    virtual ~AgentVector_CAgent();
    template <typename T>
    T getVariable(const std::string& variable_name) const;
    template <typename T, unsigned int N>
    std::array<T, N> getVariable(const std::string& variable_name) const;
    template <typename T>
    T getVariable(const std::string& variable_name, const unsigned int& index) const;
#ifdef SWIG
    template <typename T>
    std::vector<T> getVariableArray(const std::string& variable_name) const;
#endif

 protected:
    /**
     * Constructor, only ever called by AgentVector
     */
    AgentVector_CAgent(const std::shared_ptr<const AgentData> &agent, const std::weak_ptr<AgentVector::AgentDataMap> &data, AgentVector::size_type pos);
    /**
     * Index within _data
     */
    const unsigned int index;
    /**
     * Data store
     */
    const std::weak_ptr<AgentVector::AgentDataMap> _data;
    /**
     * Copy of agent definition
     */
    std::shared_ptr<const AgentData> _agent;
};

/**
 * non-const view into AgentVector
 */
class AgentVector_Agent : public AgentVector_CAgent {
    friend AgentVector::Agent AgentVector::at(AgentVector::size_type);
    friend AgentVector::Agent AgentVector::iterator::operator*() const;
    friend AgentVector::Agent AgentVector::reverse_iterator::operator*() const;

 public:
    template <typename T>
    void setVariable(const std::string &variable_name, const T &value);
    template <typename T, unsigned int N>
    void setVariable(const std::string &variable_name, const std::array<T, N> &value);
    template <typename T>
    void setVariable(const std::string &variable_name, const unsigned int &index, const T &value);
#ifdef SWIG
    template <typename T>
    void setVariableArray(const std::string &variable_name, const std::vector<T> &value);
    void setData(const AgentVector_Agent & other) {
        const auto data = _data.lock();
        if (!data) {
            THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
                "in AgentVector_Agent::setVariable().\n");
        }
        const auto other_data = other._data.lock();
        if (!other_data) {
            THROW ExpiredWeakPtr("The AgentVector which owns the passed AgentVector::Agent has been deallocated, "
                "in AgentVector_Agent::setVariable().\n");
        }
        if (_agent == other._agent || *_agent == *other._agent) {
            if (index != other.index) {
                // Copy individual members as they point to different items    const auto v_it = _data.find(variable_name);
                for (const auto &it : *data) {
                    auto& src_buff = other_data->at(it.first);
                    auto& dst_buff = it.second;
                    const char* src_ptr = static_cast<const char*>(src_buff->getReadOnlyDataPtr()) + (index * dst_buff->getVariableSize());
                    char* dest_ptr = static_cast<char*>(dst_buff->getDataPtr()) + (index * dst_buff->getVariableSize());
                    memcpy(dest_ptr, src_ptr, dst_buff->getVariableSize());
                }
            }
        } else {
            THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
                "in AgentVector_Agent::setData().\n",
                other._agent->name.c_str(), _agent->name.c_str());
        }
    }
#endif

 private:
    /**
     * Constructor, only ever called by AgentVector
     */
    AgentVector_Agent(const std::shared_ptr<const AgentData> &agent, const std::weak_ptr<AgentVector::AgentDataMap> &data, AgentVector::size_type pos);
};

template <typename T>
void AgentVector_Agent::setVariable(const std::string &variable_name, const T &value) {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::setVariable().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff->getElements() != 1) {
        THROW InvalidVarType("Variable '%s' is an array variable, use the array method instead, "
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str());
    }
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    // do the replace
    static_cast<T*>(v_buff->getDataPtr())[index] = value;
}
template <typename T, unsigned int N>
void AgentVector_Agent::setVariable(const std::string &variable_name, const std::array<T, N> &value) {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::setVariable().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff->getElements() != N) {
        THROW InvalidVarType("Variable '%s' has '%u' elements, but an array of length %u was passed, "
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str(), v_buff->getElements(), N);
    }
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    memcpy(static_cast<T*>(v_buff->getDataPtr()) + (index * N), value.data(), sizeof(T) * N);
}
template <typename T>
void AgentVector_Agent::setVariable(const std::string &variable_name, const unsigned int &array_index, const T &value) {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
          "in AgentVector_Agent::setVariable().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::setVariable().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    if (array_index >= v_buff->getElements()) {
        THROW OutOfBoundsException("Index '%u' exceeds array bounds [0-%u) of variable '%s',  "
            "in AgentVector_Agent::setVariable().",
            array_index, v_buff->getElements(), variable_name.c_str());
    }
    static_cast<T*>(v_buff->getDataPtr())[(index * v_buff->getElements()) + array_index] = value;
}
#ifdef SWIG
template <typename T>
void AgentVector_Agent::setVariableArray(const std::string &variable_name, const std::vector<T> &value) {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::setVariableArray().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::setVariableArray().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff->getElements() != value.size()) {
        THROW InvalidVarType("Variable '%s' has '%u' elements, but an array of length %u was passed, "
            "in AgentVector_Agent::setVariableArray().",
            variable_name.c_str(), v_buff->getElements(), value.size());
    }
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::setVariableArray().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    memcpy(static_cast<T*>(v_buff->getDataPtr()) + (index * v_buff->getElements()), value.data(), sizeof(T) * v_buff->getElements());
}
#endif

template <typename T>
T AgentVector_CAgent::getVariable(const std::string &variable_name) const {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::getVariable().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str());
    }
    const auto &v_buff = v_it->second;
    if (v_buff->getElements() != 1) {
        THROW InvalidVarType("Variable '%s' is an array variable, use the array method instead, "
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str());
    }
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    return static_cast<const T*>(v_buff->getReadOnlyDataPtr())[index];
}
template <typename T, unsigned int N>
std::array<T, N> AgentVector_CAgent::getVariable(const std::string &variable_name) const {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::getVariable().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (v_buff->getElements() != N) {
        THROW InvalidVarType("Variable '%s' has '%u' elements, but an array of length %u was passed, "
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str(), v_buff->getElements(), N);
    }
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    std::array<T, N> rtn;
    memcpy(rtn.data(), static_cast<const T*>(v_buff->getReadOnlyDataPtr()) + (index * N), sizeof(T) * N);
    return rtn;
}
template <typename T>
T AgentVector_CAgent::getVariable(const std::string &variable_name, const unsigned int &array_index) const {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::getVariable().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (array_index >= v_buff->getElements()) {
        THROW OutOfBoundsException("Index '%u' exceeds array bounds [0-%u) of variable '%s',  "
            "in AgentVector_Agent::getVariable().",
            array_index, v_buff->getElements(), variable_name.c_str());
    }
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::getVariable().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    return static_cast<const T*>(v_buff->getReadOnlyDataPtr())[(index * v_buff->getElements()) + array_index];
}
#ifdef SWIG
template <typename T>
std::vector<T> AgentVector_CAgent::getVariableArray(const std::string& variable_name) const {
    const auto data = _data.lock();
    if (!data) {
        THROW ExpiredWeakPtr("The AgentVector which owns this AgentVector::Agent has been deallocated, "
            "in AgentVector_Agent::getVariableArray().\n");
    }
    const auto v_it = data->find(variable_name);
    if (v_it == data->end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentVector_Agent::getVariableArray().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (v_buff->getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentVector_Agent::getVariableArray().",
            variable_name.c_str(), v_buff->getType().name(), typeid(T).name());
    }
    std::vector<T> rtn(static_cast<size_t>(v_buff->getElements()));
    memcpy(rtn.data(), static_cast<T*>(v_buff->getDataPtr()) + (index * v_buff->getElements()), sizeof(T) * v_buff->getElements());
    return rtn;
}
#endif

#endif  // INCLUDE_FLAMEGPU_POP_AGENTVECTOR_AGENT_H_
