#ifndef INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_
#define INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_

#include <memory>
#include <map>
#include <string>
#include <vector>


#include "flamegpu/model/AgentData.h"
#include "flamegpu/pop/AgentVector.h"

namespace flamegpu {

class AgentDescription;

/**
 * This class represents standalone copy of a single agent
 * It stores all the data internally, unlike an AgentVector::Agent
 * @note Not 100% on the name, might change
 */
class AgentInstance {
    friend AgentVector::iterator AgentVector::insert(AgentVector::const_iterator pos, AgentVector::size_type count, const AgentInstance& value);
    friend AgentVector::iterator AgentVector::insert(AgentVector::size_type pos, AgentVector::size_type count, const AgentInstance& value);

 public:
    /**
     * Initialises the agent variables with their default values
     */
    explicit AgentInstance(const AgentDescription &agent_desc);

    /**
     * Copy constructors
     */
    AgentInstance(const AgentInstance& other);
    explicit AgentInstance(const AgentVector::CAgent& other);

    /**
     * Move constructor
     */
    AgentInstance(AgentInstance&& other) noexcept;

    /**
     * Assignment operators
     */
    AgentInstance& operator=(const AgentInstance& other);
    AgentInstance& operator=(const AgentVector::CAgent& other);
    AgentInstance& operator=(AgentInstance&& other) noexcept;

    /**
     * Getters
     */
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

    /**
     * Setters
     */
    template <typename T>
    void setVariable(const std::string& variable_name, const T& value);
    template <typename T, unsigned int N>
    void setVariable(const std::string& variable_name, const std::array<T, N>& value);
    template <typename T>
    void setVariable(const std::string& variable_name, const unsigned int& index, const T& value);
#ifdef SWIG
    template <typename T>
    void setVariableArray(const std::string& variable_name, const std::vector<T>& value);
#endif

 private:
    std::map<std::string, util::Any> _data;
    std::shared_ptr<const AgentData> _agent;
};


template <typename T>
T AgentInstance::getVariable(const std::string& variable_name) const {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::getVariable().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (v_buff.elements != 1) {
        THROW InvalidVarType("Variable '%s' is an array variable, use the array method instead, "
            "in AgentInstance::getVariable().",
            variable_name.c_str());
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::getVariable().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    return *static_cast<const T*>(v_buff.ptr);
}
template <typename T, unsigned int N>
std::array<T, N> AgentInstance::getVariable(const std::string& variable_name) const {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::getVariable().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (v_buff.elements != N) {
        THROW InvalidVarType("Variable '%s' has '%u' elements, but an array of length %u was passed, "
            "in AgentInstance::getVariable().",
            variable_name.c_str(), v_buff.elements, N);
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::getVariable().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    std::array<T, N> rtn;
    memcpy(rtn.data(), v_buff.ptr, sizeof(T) * N);
    return rtn;
}
template <typename T>
T AgentInstance::getVariable(const std::string& variable_name, const unsigned int& index) const {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::getVariable().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (index >= v_buff.elements) {
        THROW OutOfBoundsException("Index '%u' exceeds array bounds [0-%u) of variable '%s',  "
            "in AgentInstance::getVariable().",
            index, v_buff.elements, variable_name.c_str());
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::getVariable().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    return static_cast<T*>(v_buff.ptr)[index];
}
#ifdef SWIG
template <typename T>
std::vector<T> AgentInstance::getVariableArray(const std::string& variable_name) const {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::getVariableArray().",
            variable_name.c_str());
    }
    const auto& v_buff = v_it->second;
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::getVariableArray().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    std::vector<T> rtn(static_cast<size_t>(v_buff.elements));
    memcpy(rtn.data(), static_cast<T*>(v_buff.ptr), sizeof(T) * v_buff.elements);
    return rtn;
}
#endif
template <typename T>
void AgentInstance::setVariable(const std::string& variable_name, const T& value) {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::setVariable().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff.elements != 1) {
        THROW InvalidVarType("Variable '%s' is an array variable, use the array method instead, "
            "in AgentInstance::setVariable().",
            variable_name.c_str());
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::setVariable().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    // do the replace
    *static_cast<T*>(v_buff.ptr) = value;
}
template <typename T, unsigned int N>
void AgentInstance::setVariable(const std::string& variable_name, const std::array<T, N>& value) {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::setVariable().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff.elements != N) {
        THROW InvalidVarType("Variable '%s' has '%u' elements, but an array of length %u was passed, "
            "in AgentInstance::setVariable().",
            variable_name.c_str(), v_buff.elements, N);
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::setVariable().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    memcpy(static_cast<T*>(v_buff.ptr), value.data(), sizeof(T) * N);
}
template <typename T>
void AgentInstance::setVariable(const std::string& variable_name, const unsigned int& index, const T& value) {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::setVariable().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (index >= v_buff.elements) {
        THROW OutOfBoundsException("Index '%u' exceeds array bounds [0-%u) of variable '%s',  "
            "in AgentInstance::setVariable().",
            index, v_buff.elements, variable_name.c_str());
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::setVariable().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    static_cast<T*>(v_buff.ptr)[index] = value;
}
#ifdef SWIG
template <typename T>
void AgentInstance::setVariableArray(const std::string& variable_name, const std::vector<T>& value) {
    const auto v_it = _data.find(variable_name);
    if (v_it == _data.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent, "
            "in AgentInstance::setVariableArray().",
            variable_name.c_str());
    }
    auto& v_buff = v_it->second;
    if (v_buff.elements != value.size()) {
        THROW InvalidVarType("Variable '%s' has '%u' elements, but an array of length %u was passed, "
            "in AgentInstance::setVariableArray().",
            variable_name.c_str(), v_buff.elements, value.size());
    }
    if (v_buff.type != std::type_index(typeid(T))) {
        THROW InvalidVarType("Variable '%s' is of a different type. "
            "'%s' was expected, but '%s' was requested,"
            "in AgentInstance::setVariableArray().",
            variable_name.c_str(), v_buff.type.name(), typeid(T).name());
    }
    memcpy(static_cast<T*>(v_buff.ptr), value.data(), sizeof(T) * v_buff.elements);
}
#endif  // SWIG

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_
