#ifndef INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_

#include <utility>
#include <string>
#include <set>
#include <map>

#include "ModelDescription.h"

//TODO:
class FLAMEGPU_HOST_API{};
enum FLAME_GPU_SUBMODEL_EXIT_STATUS { CONTINUE, EXIT };
typedef FLAME_GPU_SUBMODEL_EXIT_STATUS(*FLAMEGPU_SUBMODEL_EXIT_FUNCTION_POINTER)(FLAMEGPU_HOST_API *api);

#define FLAMEGPU_SUBMODEL_EXIT_CONDITION(funcName) \
FLAME_GPU_SUBMODEL_EXIT_STATUS funcName ## _impl(FLAMEGPU_HOST_API* FLAMEGPU); \
FLAMEGPU_SUBMODEL_EXIT_FUNCTION_POINTER funcName = funcName ## _impl;\
FLAME_GPU_SUBMODEL_EXIT_STATUS funcName ## _impl(FLAMEGPU_HOST_API* FLAMEGPU)

/**
 * A representation of the elements of a model specific to a sub-model:
 * > Which agents (and their variables) are imported?
 * > Are any read-only messages imported?
 * > Are any environment variables imported?
 * > What is the end condition of the submodel?
 * @issue Unclear whether this should be an optional member to ModelDescription
 *     or a separate object either that pairs-with/encapsulates the ModelDescription
 */
class SubModelImports
{
 public:
    typedef std::map<const std::string, std::set<const std::string>> SubAgents;
    typedef std::map<const std::string, std::set<const std::string>> SubMessages;
    //TODO: typedef std::set<const std::string> SubEnvironmentConstants;

    typedef std::pair<const std::string, std::set<const std::string>> SubAgent;
    typedef std::pair<const std::string, std::set<const std::string>> SubMessage;
    //TODO: std::string SubEnvironmentConstant;

    SubModelImports(const ModelDescription &model)
        : m_model(model)
    {

    }
    /**
     * Mark one of the model's agents as having imported variables
     * @param name The name of the agent within the model
     * @param variable_names The name of the agent variables to be imported
     * @note imported variables will be merged with any existing imports for the same agent
     */
    void addAgentImport(const std::string &name, const std::set<const std::string> &variable_names) {
        // Check the agent and variables exist in the model
        {
            const AgentMap &parent_agents = m_model.getAgentMap();
            const auto &parent_agent = parent_agents.find(name);
            if (parent_agent != parent_agents.end()) {
                // SubAgent exists
                const MemoryMap &parent_agent_vars = parent_agent->second.getMemoryMap();
                for (auto &&s_agent_var : variable_names) {
                    const auto &parent_agent_var = parent_agent_vars.find(s_agent_var);
                    if (parent_agent_var != parent_agent_vars.end())
                        throw InvalidAgentVar();
                }
            }
            else
                throw InvalidCudaAgent();
        }

        // If we got this far, it validated!
        const auto &agentImport = m_agentImports.find(name);
        if (agentImport != m_agentImports.end()) {
            //Merge with existing SubAgent
            agentImport->second.insert(variable_names.begin(), variable_names.end());
        } else {
            //New SubAgent
            m_agentImports.insert({ name, variable_names });
        }
    }
    /**
     * Mark one of the model's messages as having imported variables
     * @param name The name of the message within the model
     * @param variable_names The name of the message variables to be imported
     * @note imported variables will be merged with any existing imports for the same agent
     */
    void addMessageImport(const std::string &name, const std::set<const std::string> &variable_names) {
        // Check the agent and variables exist in the model
        {
            const MessageMap &parent_messages = m_model.getMessageMap();
            const auto &parent_Message = parent_messages.find(name);
            if (parent_Message != parent_messages.end()) {
                // SubMessage exists
                const MemoryMap &parent_message_vars = parent_Message->second.getVariableMap();
                for (auto &&s_message_var : variable_names) {
                    const auto &parent_message_var = parent_message_vars.find(s_message_var);
                    if (parent_message_var != parent_message_vars.end())
                        throw InvalidMessageVar();
                }
            }
            else
                throw InvalidCudaMessage();
        }

        //TODO: Validate that model does not write to the messages??

        // If we got this far, it validated!
        const auto &messageImport = m_messageImports.find(name);
        if (messageImport != m_messageImports.end()) {
            //Merge with existing SubAgent
            messageImport->second.insert(variable_names.begin(), variable_names.end());
        }
        else {
            //New SubAgent
            m_messageImports.insert({ name, variable_names });
        }
    }
    //TODO: void addEnvironmentConstantImport(const std::string &name) { }

    //TODO: Utility methods, add all of an agent/message's variables
    //TODO: Utility methods, add agent/message's variables individually
    /**
    * Name of the SubModelExitConditionFunction (Step function with bool return)
    */
    void setExitCondition(FLAMEGPU_SUBMODEL_EXIT_FUNCTION_POINTER *func_p) {
        m_exitCondition = func_p;
    }

 private:
    /**
     * Tests that every stored identifier exists within the model (m_model)
     * @return Returns true if no identifiers are missing
     * @note This should be redundant, as they should be checked on creation
     */
    bool validate() {
        unsigned int errCount = 0;
        {
            // SubAgents
            const AgentMap &parent_agents = m_model.getAgentMap();
            for (auto &&s_agent : m_agentImports) {
                const auto &parent_agent = parent_agents.find(s_agent.first);
                if(parent_agent != parent_agents.end()) {
                    // SubAgent exists!
                    const MemoryMap &parent_agent_vars = parent_agent->second.getMemoryMap();
                    for (auto &&s_agent_var : s_agent.second) {
                        const auto &parent_agent_var = parent_agent_vars.find(s_agent_var);
                        if (parent_agent_var != parent_agent_vars.end())
                            errCount++; //SubAgentVar missing!
                    }
                }
                else
                    errCount++;  //SubAgent missing!
            }
        }
        {
            // SubMessages
            const MessageMap &parent_messages = m_model.getMessageMap();
            for (auto &&s_message : m_messageImports) {
                const auto &parent_message = parent_messages.find(s_message.first);
                if (parent_message != parent_messages.end()) {
                    // SubAgent exists!
                    const auto &parent_message_vars = parent_message->second.getVariableMap();
                    for (auto &&s_message_var : s_message.second) {
                        const auto &parent_message_var = parent_message_vars.find(s_message_var);
                        if (parent_message_var != parent_message_vars.end())
                            errCount++; //SubMessageVar missing!
                    }
                }
                else
                    errCount++;  //SubMessage missing!
            }
        }
        {
            // SubEnvironmentConstants
            //TODO: EnvConstants don't yet exist
        }
        return errCount==0;
    }
    /**
     * The model which this class describes imports for
     */
    const ModelDescription &m_model;
    /**
     * Mapping of agent name -> importable agent variables
     */
    SubAgents m_agentImports;
    /**
     * Mapping of message name -> importable message variables
     */
    SubMessages m_messageImports;
    /**
     * List of importable environment constants
     */
    //TODO: SubEnvironmentConstants m_environmentConstantImports;
    /**
     * Pointer to the submodel exit function executed to check whether to exit
     */
    FLAMEGPU_SUBMODEL_EXIT_FUNCTION_POINTER *m_exitCondition = nullptr;
};

#endif // INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_