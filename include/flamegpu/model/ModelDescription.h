#ifndef INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_

#include <string>
#include <map>
#include <memory>

#include "flamegpu/model/ModelData.h"

class AgentDescription;
class MessageDescription;

class ModelDescription {

 public:
    /**
     * Constructors
     */
    ModelDescription();
    ModelDescription(const std::string &model_name);
    // Copy Construct
    ModelDescription(const ModelDescription &other_model);
    // Move Construct
    ModelDescription(ModelDescription &&other_model);
    // Copy Assign
    ModelDescription& operator=(const ModelDescription &other_model);
    // Move Assign
    ModelDescription& operator=(ModelDescription &&other_model);

    /**
     * Accessors
     */    
    AgentDescription& newAgent(const std::string &agent_name);
    AgentDescription& Agent(const std::string &agent_name);
    AgentDescription& cloneAgent(const AgentDescription &agent);
    
    MessageDescription& newMessage(const std::string &message_name);
    MessageDescription& Message(const std::string &message_name);
    MessageDescription& cloneMessage(const MessageDescription &message);
    
    EnvironmentDescription& Environment();
    EnvironmentDescription& cloneEnvironment(const EnvironmentDescription &environment);

    /**
     * Const Accessors
     */
    std::string getName() const;
    
    const AgentDescription& getAgent(const std::string &agent_name) const;
    const MessageDescription& getMessage(const std::string &message_name) const;
    const EnvironmentDescription& getEnvironment() const;

    bool hasAgent(const std::string &agent_name) const;
    bool hasMessage(const std::string &message_name) const;

    ModelDescription clone(const std::string &cloned_model_name) const;

 private:
     std::shared_ptr<ModelData> model;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_MODELDESCRIPTION_H_
