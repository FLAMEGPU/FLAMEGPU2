#ifndef INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_

#include <string>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

struct ModelData;
struct MessageData;

/**
 * Within the model hierarchy, this class represents the definition of an message for a FLAMEGPU model
 * This class is used to configure external elements of messages, such as variables
 * Base-class, represents brute-force messages
 * Can be extended by more advanced message descriptors
 * @see MessageData The internal data store for this class
 * @see ModelDescription::newMessage(const std::string&) For creating instances of this class
 */
class MessageDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct MessageData;
    friend void AgentFunctionDescription::setMessageOutput(MessageDescription&);
    friend void AgentFunctionDescription::setMessageInput(MessageDescription&);

 protected:
    /**
     * Constructors
     */
    MessageDescription(ModelData *const _model, MessageData *const data);
    /**
     * Default copy constructor, not implemented
     */
    MessageDescription(const MessageDescription &other_message) = delete;
    /**
     * Default move constructor, not implemented
     */
    MessageDescription(MessageDescription &&other_message) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    MessageDescription& operator=(const MessageDescription &other_message) = delete;
    /**
     * Default move assignment, not implemented
     */
    MessageDescription& operator=(MessageDescription &&other_message) noexcept = delete;

 public:
    /**
     * Equality operator, checks whether MessageDescription hierarchies are functionally the same
     * @returns True when messages are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const MessageDescription& rhs) const;
    /**
     * Equality operator, checks whether MessageDescription hierarchies are functionally different
     * @returns True when messages are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const MessageDescription& rhs) const;

    /**
     * Adds a new variable to the message
     * @param variable_name Name of the variable
     * @tparam T Type of the message variable, this must be an arithmetic type
     * @tparam N The length of the variable array (1 if not an array, must be greater than 0)
     * @throws InvalidAgentVar If a variable already exists within the message with the same name
     * @throws InvalidAgentVar If N is <= 0
     */
    template<typename T, ModelData::size_type N = 1>
    void newVariable(const std::string &variable_name);

    /**
     * @return The message's name
     */
    std::string getName() const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The type of the named variable
     * @throws InvalidAgentVar If a variable with the name does not exist within the message
     */
    std::type_index getVariableType(const std::string &variable_name) const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The size of the named variable's type
     * @throws InvalidAgentVar If a variable with the name does not exist within the message
     */
    size_t getVariableSize(const std::string &variable_name) const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The number of elements in the name variable (1 if it isn't an array)
     * @throws InvalidAgentVar If a variable with the name does not exist within the message
     */
    ModelData::size_type getVariableLength(const std::string &variable_name) const;
    /**
     * @return The total number of variables within the message
     */
    ModelData::size_type getVariablesCount() const;
    /**
     * @param variable_name Name of the variable to check
     * @return True when a variable with the specified name exists within the message
     */
    bool hasVariable(const std::string &variable_name) const;

 protected:
    /**
     * Root of the model hierarchy
     */
    ModelData *const model;
    /**
     * The class which stores all of the message's data.
     */
    MessageData *const message;
};

class Spatial2DMessageDescription : public MessageDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct Spatial2DMessageData;

 protected:
    /**
     * Constructors
     */
    Spatial2DMessageDescription(ModelData *const _model, Spatial2DMessageData *const data);
    /**
     * Default copy constructor, not implemented
     */
    Spatial2DMessageDescription(const Spatial2DMessageDescription &other_message) = delete;
    /**
     * Default move constructor, not implemented
     */
    Spatial2DMessageDescription(Spatial2DMessageDescription &&other_message) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    Spatial2DMessageDescription& operator=(const Spatial2DMessageDescription &other_message) = delete;
    /**
     * Default move assignment, not implemented
     */
    Spatial2DMessageDescription& operator=(Spatial2DMessageDescription &&other_message) noexcept = delete;

 public:
    void setRadius(const float &r);
    void setMinX(const float &x);
    void setMinY(const float &y);
    void setMin(const float &x, const float &y);
    void setMaxX(const float &x);
    void setMaxY(const float &y);
    void setMax(const float &x, const float &y);

    float &Radius();
    float &MinX();
    float &MinY();
    float &MaxX();
    float &MaxY();

    float getRadius() const;
    float getMinX() const;
    float getMinY() const;
    float getMaxX() const;
    float getMaxY() const;
};

class Spatial3DMessageDescription : protected Spatial2DMessageDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct Spatial3DMessageData;

 protected:
    /**
     * Constructors
     */
    Spatial3DMessageDescription(ModelData *const _model, Spatial3DMessageData *const data);
    /**
     * Default copy constructor, not implemented
     */
    Spatial3DMessageDescription(const Spatial3DMessageDescription &other_message) = delete;
    /**
     * Default move constructor, not implemented
     */
    Spatial3DMessageDescription(Spatial3DMessageDescription &&other_message) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    Spatial3DMessageDescription& operator=(const Spatial3DMessageDescription &other_message) = delete;
    /**
     * Default move assignment, not implemented
     */
    Spatial3DMessageDescription& operator=(Spatial3DMessageDescription &&other_message) noexcept = delete;

 public:
    using Spatial2DMessageDescription::setRadius;
    using Spatial2DMessageDescription::setMinX;
    using Spatial2DMessageDescription::setMinY;
    void setMinZ(const float &z);
    void setMin(const float &x, const float &y, const float &z);
    using Spatial2DMessageDescription::setMaxX;
    using Spatial2DMessageDescription::setMaxY;
    void setMaxZ(const float &z);
    void setMax(const float &x, const float &y, const float &z);

    using Spatial2DMessageDescription::Radius;
    using Spatial2DMessageDescription::MinX;
    using Spatial2DMessageDescription::MinY;
    float &MinZ();
    using Spatial2DMessageDescription::MaxX;
    using Spatial2DMessageDescription::MaxY;
    float &MaxZ();
    using Spatial2DMessageDescription::getRadius;
    using Spatial2DMessageDescription::getMinX;
    using Spatial2DMessageDescription::getMinY;
    float getMinZ() const;
    using Spatial2DMessageDescription::getMaxX;
    using Spatial2DMessageDescription::getMaxY;
    float getMaxZ() const;
};

/**
 * Template implementation
 */
template<typename T, ModelData::size_type N>
void MessageDescription::newVariable(const std::string &variable_name) {
    // Array length 0 makes no sense
    static_assert(N > 0, "A variable cannot have 0 elements.");
    if (message->variables.find(variable_name) == message->variables.end()) {
        message->variables.emplace(variable_name, ModelData::Variable(N, T()));
        return;
    }
    THROW InvalidMessageVar("Message ('%s') already contains variable '%s', "
        "in MessageDescription::newVariable().",
        message->name.c_str(), variable_name.c_str());
}

#endif  // INCLUDE_FLAMEGPU_MODEL_MESSAGEDESCRIPTION_H_
