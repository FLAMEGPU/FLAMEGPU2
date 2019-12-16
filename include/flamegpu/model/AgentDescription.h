#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_

#include <string>
#include <map>
#include <typeinfo>
#include <memory>
#include <vector>
#include <set>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
class AgentFunctionDescription;
class MessageDescription;
struct ModelData;
struct AgentData;

class AgentDescription {
    friend struct AgentData;
    friend struct AgentFunctionData;
    friend AgentPopulation::AgentPopulation(const AgentDescription &, unsigned int);

    /**
     * Only way to construct an AgentDescription
     */
    friend AgentDescription& ModelDescription::newAgent(const std::string &);

    /**
     * Constructors
     */
    AgentDescription(ModelData *const _model, AgentData *const data);
    // Copy Construct
    AgentDescription(const AgentDescription &other_agent);
    // Move Construct
    AgentDescription(AgentDescription &&other_agent) noexcept;
    // Copy Assign
    AgentDescription& operator=(const AgentDescription &other_agent) = delete;
    // Move Assign
    AgentDescription& operator=(AgentDescription &&other_agent) noexcept = delete;

 public:
    /**
     * Accessors
     */
    void newState(const std::string &state_name);
    void setInitialState(const std::string &initial_state);

    template<typename AgentVariable, ModelData::size_type N = 1>
    void newVariable(const std::string &variable_name);

    template<typename AgentFunction>
    AgentFunctionDescription &newFunction(const std::string &function_name, AgentFunction a = AgentFunction());
    AgentFunctionDescription &Function(const std::string &function_name);
    AgentFunctionDescription &cloneFunction(const AgentFunctionDescription &function);

    /**
     * Const Accessors
     */
    std::string getName() const;

    std::type_index getVariableType(const std::string &variable_name) const;
    size_t getVariableSize(const std::string &variable_name) const;
    ModelData::size_type getVariableLength(const std::string &variable_name) const;
    ModelData::size_type getVariablesCount() const;
    const AgentFunctionDescription& getFunction(const std::string &function_name) const;

    bool hasState(const std::string &state_name) const;
    bool hasVariable(const std::string &variable_name) const;
    bool hasFunction(const std::string &function_name) const;
    bool isOutputOnDevice() const;

    const std::set<std::string> &getStates() const;

 private:
    ModelData *const model;
    AgentData *const agent;
};

/**
 * Template implementation
 */
template <typename T, ModelData::size_type N>
void AgentDescription::newVariable(const std::string &variable_name) {
    if (agent->variables.find(variable_name) == agent->variables.end()) {
        agent->variables.emplace(variable_name, ModelData::Variable(N, T()));
        return;
    }
    THROW InvalidAgentVar("Agent ('%s') already contains variable '%s', "
        "in AgentDescription::newVariable().",
        agent->name.c_str(), variable_name.c_str());
}

// Found in "flamegpu/model/AgentFunctionDescription.h"
// template<typename AgentFunction>
// AgentFunctionDescription &AgentDescription::newFunction(const std::string &function_name, AgentFunction)

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
