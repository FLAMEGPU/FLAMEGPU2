#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_

#ifndef __CUDACC_RTC__
#include <typeindex>
#include <memory>
#include <unordered_map>
#include <string>


#include "flamegpu/model/Variable.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#endif  // __CUDACC_RTC__
#include "flamegpu/runtime/messaging/None.h"



struct ModelData;

/**
 * Brute force messaging functionality
 *
 * Every agent accesses all messages
 * This technique is expensive, and other techniques are preferable if operating with more than 1000 messages.
 */
class MsgBruteForce {
 public:
    /**
     * Common size type
     */
    typedef MsgNone::size_type size_type;

    class Message;      // Forward declare inner classes
    class iterator;     // Forward declare inner classes
    struct Data;        // Forward declare inner classes
    class Description;  // Forward declare inner classes
    /**
     * MetaData required by brute force during message reads
     */
    struct MetaData {
        unsigned int length;
    };
    /**
     * This class is accessible via FLAMEGPU_DEVICE_API.message_in if MsgBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for reading brute force messages
     */
    class In {
        /**
         * Message has full access to In, they are treated as the same class so share everything
         * Reduces/memory data duplication
         */
        friend class MsgBruteForce::Message;

     public:
        /**
         * Constructer
         * Initialises member variables
         * @param agentfn_hash Added to msg_hash to produce combined_hash
         * @param msg_hash Added to agentfn_hash to produce combined_hash
         * @param metadata Reinterpreted as type MsgBruteForce::MetaData to extract length
         */
        __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *metadata)
            : combined_hash(agentfn_hash + msg_hash)
            , len(reinterpret_cast<const MetaData*>(metadata)->length)
        { }
        /**
         * Returns the number of elements in the message list.
         */
        __device__ size_type size(void) const {
            return len;
        }
        /**
          * Returns an iterator to the start of the message list
          */
        __device__ iterator begin(void) const {  // const
            return iterator(*this, 0);
        }
        /**
         * Returns an iterator to the position beyond the end of the message list
         */
        __device__ iterator end(void) const  {  // const
            // If there can be many begin, each with diff end, we need a middle layer to host the iterator/s
            return iterator(*this, len);
        }

     private:
         /**
          * CURVE hash for accessing message data
          * agent function hash + message hash
          */
        Curve::NamespaceHash combined_hash;
        /**
         * Total number of messages in the message list
         */
        size_type len;
    };
    /**
     * Provides access to a specific message
     * Returned by the iterator
     * @see In::iterator
     */
    class Message {
         /**
          * Paired In class which created the iterator
          */
        const MsgBruteForce::In &_parent;
        /**
         * Position within the message list
         */
        size_type index;

     public:
        /**
         * Constructs a message and directly initialises all of it's member variables
         * index is always init to 0
         * @note See member variable documentation for their purposes
         */
        __device__ Message(const MsgBruteForce::In &parent) : _parent(parent), index(0) {}
        /**
         * Alternate constructor, allows index to be manually set
         * @note I think this is unused
         */
        __device__ Message(const MsgBruteForce::In &parent, size_type index) : _parent(parent), index(index) {}
        /**
         * Equality operator
         * Compares all internal member vars for equality
         * @note Does not compare _parent
         */
        __host__ __device__ bool operator==(const Message& rhs) const { return  this->getIndex() == rhs.getIndex(); }
        /**
         * Inequality operator
         * Returns inverse of equality operator
         * @see operator==(const Message&)
         */
        __host__ __device__ bool operator!=(const Message& rhs) const { return  this->getIndex() != rhs.getIndex(); }
        /**
         * Updates the message to return variables from the next message in the message list
         * @return Returns itself
         */
        __host__ __device__ Message& operator++() { ++index;  return *this; }
        /**
         * Returns the index of the message within the full message list
         */
        __host__ __device__ size_type getIndex() const { return this->index; }
        /**
         * Returns the value for the current message attached to the named variable
         * @param variable_name Name of the variable
         * @tparam T type of the variable
         * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
         * @return The specified variable, else 0x0 if an error occurs
         */
        template<typename T, size_type N>
        __device__ T getVariable(const char(&variable_name)[N]) const;
    };
    /**
    * Stock iterator for iterating MsgBruteForce::In::Message objects
    */
    class iterator{  // : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
        /**
         * The message returned to the user
         */
        Message _message;

     public:
        /**
         * Constructor
         * This iterator is constructed by MsgBruteForce::begin()
         * @see MsgBruteForce::begin()
         */
        __device__ iterator(const In &parent, size_type index) : _message(parent, index) {}
        /**
         * Moves to the next message
         */
        __device__ iterator& operator++() { ++_message;  return *this; }
        /**
         * Equality operator
         * Compares message
         */
        __device__ bool operator==(const iterator& rhs) const { return  _message == rhs._message; }
        /**
         * Inequality operator
         * Compares message
         */
        __device__ bool operator!=(const iterator& rhs) const { return  _message != rhs._message; }
        /**
         * Dereferences the iterator to return the message object, for accessing variables
         */
        __device__ Message& operator*() { return _message; }
    };

    /**
     * This class is accessible via FLAMEGPU_DEVICE_API.message_out if MsgBruteForce is specified in FLAMEGPU_AGENT_FUNCTION
     * It gives access to functionality for outputting brute force messages
     */
    class Out {
     public:
        /**
         * Constructer
         * Initialises member variables
         * @param agentfn_hash Added to msg_hash to produce combined_hash
         * @param msg_hash Added to agentfn_hash to produce combined_hash
         * @param _streamId Stream index, used for optional message output flag array
         */
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int _streamId)
            : combined_hash(agentfn_hash + msg_hash)
            , streamId(_streamId)
        { }
        /**
         * Sets the specified variable for this agents message
         * @param variable_name Name of the variable
         * @tparam T type of the variable
         * @tparam N Length of variable name (this should be implicit if a string literal is passed to variable name)
         * @return The specified variable, else 0x0 if an error occurs
         */
        template<typename T, unsigned int N>
        __device__ void setVariable(const char(&variable_name)[N], T value) const;

     protected:
        /**
         * CURVE hash for accessing message data
         * agentfn_hash + msg_hash
         */
        Curve::NamespaceHash combined_hash;
        /**
         * Stream index used for setting optional message output flag
         */
        unsigned int streamId;
    };
#ifndef __CUDACC_RTC__
    /**
     * Blank handler, brute force requires no index or special allocations
     * Only stores the length on device
     */
    class CUDAModelHandler : public MsgSpecialisationHandler {
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
         * Updates the length of the messagelist stored on device
         */
        void buildIndex() override;
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
    struct Data {
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
        virtual Data *clone(ModelData *const newParent);
        /**
         * Copy constructor
         * This is unsafe, should only be used internally, use clone() instead
         */
        Data(ModelData *const, const Data &other);
        /**
         * Normal constructor, only to be called by ModelDescription
         */
        Data(ModelData *const, const std::string &message_name);
    };

    /**
     * Within the model hierarchy, this class represents the definition of an message for a FLAMEGPU model
     * This class is used to configure external elements of messages, such as variables
     * Base-class, represents brute-force messages
     * Can be extended by more advanced message descriptors
     * @see MessageData The internal data store for this class
     * @see ModelDescription::newMessage(const std::string&) For creating instances of this class
     */
    class Description {
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
         Description(ModelData *const _model, Data *const data);
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
         * @tparam N The length of the variable array (1 if not an array, must be greater than 0)
         * @throws InvalidAgentVar If a variable already exists within the message with the same name
         * @throws InvalidAgentVar If N is <= 0
         */
        template<typename T, size_type N = 1>
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
        size_type getVariableLength(const std::string &variable_name) const;
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
        ModelData *const model;
        /**
         * The class which stores all of the message's data.
         */
        Data *const message;
    };
#endif  // __CUDACC_RTC__
};
template<typename T, unsigned int N>
__device__ T MsgBruteForce::Message::getVariable(const char(&variable_name)[N]) const {
    // Ensure that the message is within bounds.
    if (index < this->_parent.len) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

/**
* \brief adds a message
* \param variable_name Name of message variable to set
* \param value Value to set it to
*/
template<typename T, unsigned int N>
__device__ void MsgBruteForce::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    Curve::setVariable<T>(variable_name, combined_hash, value, index);

    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_configs[flamegpu_internal::CUDAScanCompaction::MESSAGE_OUTPUT][streamId].scan_flag[index] = 1;
}
#ifndef __CUDACC_RTC__
/**
 * Template implementation
 */
template<typename T, MsgBruteForce::size_type N>
void MsgBruteForce::Description::newVariable(const std::string &variable_name) {
    // Array length 0 makes no sense
    static_assert(N > 0, "A variable cannot have 0 elements.");
    if (message->variables.find(variable_name) == message->variables.end()) {
        message->variables.emplace(variable_name, Variable(N, T()));
        return;
    }
    THROW InvalidMessageVar("Message ('%s') already contains variable '%s', "
        "in MessageDescription::newVariable().",
        message->name.c_str(), variable_name.c_str());
}

#endif  // __CUDACC_RTC__
#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_
