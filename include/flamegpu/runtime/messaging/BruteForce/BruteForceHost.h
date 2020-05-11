#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_BRUTEFORCEHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_BRUTEFORCEHOST_H_

// TODO: This should *not* be required in a .h file. Need to separate concerns between c++ and CUDA.
#include <device_launch_parameters.h>

#include <typeindex>
#include <memory>
#include <unordered_map>
#include <string>

#include "flamegpu/model/Variable.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/runtime/cuRVE/curve.h"

#include "flamegpu/runtime/messaging/None/NoneHost.h"
#include "flamegpu/runtime/messaging/BruteForce.h"


/**
 * Blank handler, brute force requires no index or special allocations
 * Only stores the length on device
 */
class MsgBruteForce::CUDAModelHandler : public MsgSpecialisationHandler {
 public:
    /**
     * Constructor
     * Allocates memory on device for message list length
     * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
     */
    explicit CUDAModelHandler(CUDAMessage &a)
        : MsgSpecialisationHandler()
        , d_metadata(nullptr)
        , sim_message(a) { }

    /** 
     * Destructor.
     * Should free any local host memory (device memory cannot be freed in destructors)
     */
    ~CUDAModelHandler() { }
    /**
     * Allocates memory for the constructed index.
     * Sets data asthough message list is empty
     * @param scatter Scatter instance and scan arrays to be used (CUDAAgentModel::singletons->scatter)
     * @param streamId Index of stream specific structures used
     */
    void init(CUDAScatter &scatter, const unsigned int &streamId) override;
    /**
     * Updates the length of the messagelist stored on device
     * @param scatter Scatter instance and scan arrays to be used (CUDAAgentModel::singletons->scatter)
     * @param streamId Index of stream specific structures used
     */
    void buildIndex(CUDAScatter &scatter, const unsigned int &streamId) override;
    /**
     * Allocates memory for the constructed index.
     * The memory allocation is checked by build index.
     */
    void allocateMetaDataDevicePtr() override;
    /**
     * Releases memory for the constructed index.
     */
    void freeMetaDataDevicePtr() override;
    /**
     * Returns a pointer to the metadata struct, this is required for reading the message data
     */
    const void *getMetaDataDevicePtr() const override { return d_metadata; }

 private:
    /**
     * Host copy of metadata struct (message list length)
     */
    MetaData hd_metadata;
    /**
     * Pointer to device copy of metadata struct (message list length)
     */
    MetaData *d_metadata;
    /**
     * Owning CUDAMessage, provides access to message storage etc
     */
    CUDAMessage &sim_message;
};

/**
 * This is the internal data store for MessageDescription
 * Users should only access that data stored within via an instance of MessageDescription
 */
struct MsgBruteForce::Data {
    friend class ModelDescription;
    friend struct ModelData;

    virtual ~Data();

    /**
     * Holds all of the message's variable definitions
     */
    VariableMap variables;
    /**
     * Description class which provides convenient accessors
     */
    std::unique_ptr<Description> description;
    /**
     * Name of the message, used to refer to the message in many functions
     */
    std::string name;
    /**
     * The number of functions that have optional output of this message type
     * This value is modified by AgentFunctionDescription
     */
    unsigned int optional_outputs;
    /**
     * Equality operator, checks whether MessageData hierarchies are functionally the same
     * @returns True when messages are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const Data& rhs) const;
    /**
     * Equality operator, checks whether MessageData hierarchies are functionally different
     * @returns True when messages are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const Data& rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    Data(const Data &other) = delete;

    virtual std::unique_ptr<MsgSpecialisationHandler> getSpecialisationHander(CUDAMessage &owner) const;

    /**
     * Used internally to validate that the corresponding Msg type is attached via the agent function shim.
     * @return The std::type_index of the Msg type which must be used.
     */
    virtual std::type_index getType() const;

 protected:
    virtual Data *clone(const std::shared_ptr<const ModelData> &newParent);
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    Data(const std::shared_ptr<const ModelData> &, const Data &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    Data(const std::shared_ptr<const ModelData> &, const std::string &message_name);
};

/**
 * Within the model hierarchy, this class represents the definition of an message for a FLAMEGPU model
 * This class is used to configure external elements of messages, such as variables
 * Base-class, represents brute-force messages
 * Can be extended by more advanced message descriptors
 * @see MessageData The internal data store for this class
 * @see ModelDescription::newMessage(const std::string&) For creating instances of this class
 */
class MsgBruteForce::Description {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct Data;
    friend class AgentFunctionDescription;
    // friend void AgentFunctionDescription::setMessageOutput(MsgBruteForce::Description&);
    // friend void AgentFunctionDescription::setMessageInput(MsgBruteForce::Description&);

 protected:
    /**
     * Constructors
     */
    Description(const std::shared_ptr<const ModelData> &_model, Data *const data);
    /**
     * Default copy constructor, not implemented
     */
    Description(const Description &other_message) = delete;
    /**
     * Default move constructor, not implemented
     */
    Description(Description &&other_message) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    Description& operator=(const Description &other_message) = delete;
    /**
     * Default move assignment, not implemented
     */
    Description& operator=(Description &&other_message) noexcept = delete;

 public:
    /**
     * Equality operator, checks whether MessageDescription hierarchies are functionally the same
     * @returns True when messages are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const Description& rhs) const;
    /**
     * Equality operator, checks whether MessageDescription hierarchies are functionally different
     * @returns True when messages are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const Description& rhs) const;

    /**
     * Adds a new variable to the message
     * @param variable_name Name of the variable
     * @tparam T Type of the message variable, this must be an arithmetic type
     * @throws InvalidMessageVar If a variable already exists within the message with the same name
     */
    template<typename T>
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
    const std::type_index& getVariableType(const std::string &variable_name) const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The size of the named variable's type
     * @throws InvalidAgentVar If a variable with the name does not exist within the message
     */
    size_t getVariableSize(const std::string &variable_name) const;
    /**
     * @return The total number of variables within the message
     */
    size_type getVariablesCount() const;
    /**
     * @param variable_name Name of the variable to check
     * @return True when a variable with the specified name exists within the message
     */
    bool hasVariable(const std::string &variable_name) const;

 protected:
    /**
     * Root of the model hierarchy
     */
    const std::weak_ptr<const ModelData> model;
    /**
     * The class which stores all of the message's data.
     */
    Data *const message;
};
/**
 * Template implementation
 */
template<typename T>
void MsgBruteForce::Description::newVariable(const std::string &variable_name) {
    if (!variable_name.empty() && variable_name[0] == '_') {
        THROW ReservedName("Message variable names cannot begin with '_', this is reserved for internal usage, "
            "in MsgBruteForce::Description::newVariable().");
    }
    if (message->variables.find(variable_name) == message->variables.end()) {
        message->variables.emplace(variable_name, Variable(1, T()));
        return;
    }
    THROW InvalidMessageVar("Message ('%s') already contains variable '%s', "
        "in MessageDescription::newVariable().",
        message->name.c_str(), variable_name.c_str());
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_BRUTEFORCEHOST_H_
