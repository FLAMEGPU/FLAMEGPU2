#ifndef INCLUDE_FLAMEGPU_MODEL_SUBAGENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBAGENTDESCRIPTION_H_

#include <memory>
#include <string>


namespace flamegpu {

struct ModelData;
struct SubModelData;
struct SubAgentData;

/**
 * This class provides an interface to a mapping between a parent and submodel agent
 */
class SubAgentDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct SubAgentData;
    /**
     * Constructor, this should only be called by AgentData
     * @param _model Model at root of model hierarchy
     * @param data Data store of this subagent's data
     */
    SubAgentDescription(const std::shared_ptr<const ModelData> &_model, SubAgentData *const data);
    /**
     * Default copy constructor, not implemented
     */
    SubAgentDescription(const SubAgentDescription &other_agent) = delete;
    /**
     * Default move constructor, not implemented
     */
    SubAgentDescription(SubAgentDescription &&other_agent) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    SubAgentDescription& operator=(const SubAgentDescription &other_agent) = delete;
    /**
     * Default move assignment, not implemented
     */
    SubAgentDescription& operator=(SubAgentDescription &&other_agent) noexcept = delete;

 public:
    /**
     * Links the named states between the master and sub agent
     * @param sub_state_name Name of the state in the sub models agent
     * @param master_state_name Name of the state in the master models agent
     * @throws InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws InvalidAgentState If the named state does not exist within the bound sub or master agent
     */
    void mapState(const std::string &sub_state_name, const std::string &master_state_name);
    /**
     * Links the named variables between the master and sub agent
     * These variables must have the same type and number of elements (1 unless they're an array variable)
     * @param sub_variable_name Name of the variable in the sub models agent
     * @param master_variable_name Name of the variable in the master models agent
     * @throws InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws InvalidAgentVar If the named variable does not exist within the bound sub or master agent
     * @throws InvalidAgentVar If there is a mismatch between the variables types or number of elements
     */
    void mapVariable(const std::string &sub_variable_name, const std::string &master_variable_name);
    /**
     * Returns the master agent state which has been mapped to the name subagent state
     * @param sub_state_name Name of the state in the sub agent to check
     * @return The name of the state within the master agent which is mapped
     * @throws InvalidAgentState If the sub agent state does not exist or has not been mapped yet
     */
    std::string getStateMapping(const std::string &sub_state_name);
    /**
     * Returns the master agent variable which has been mapped to the name subagent variable
     * @param sub_variable_name Name of the variable in the sub agent to check
     * @return The name of the variable within the master agent which is mapped
     * @throws InvalidAgentVar If the sub agent variable does not exist or has not been mapped yet
     */
    std::string getVariableMapping(const std::string &sub_variable_name);

 private:
    /**
     * Root of the model hierarchy
     */
    const std::weak_ptr<const ModelData> model;
    /**
     * The class which stores all of the agent's data.
     */
    SubAgentData *const data;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBAGENTDESCRIPTION_H_
