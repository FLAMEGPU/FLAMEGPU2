#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_DEVICE_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_DEVICE_API_H_

/**
 * @file flame_functions_api.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief  FLAMEGPU_API is a singleton class for the device runtime
 *
 * \todo longer description
 */

#include <cassert>

#include "flamegpu/gpu/CUDAErrorChecking.h"            // required for CUDA error handling functions
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/runtime/messagelist.h"
#include "flamegpu/runtime/utility/AgentRandom.cuh"
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"

// TODO: Some example code of the handle class and an example function
// ! FLAMEGPU_API is a singleton class
class FLAMEGPU_DEVICE_API;  // Forward declaration (class defined below)

// ! FLAMEGPU function return type
enum FLAME_GPU_AGENT_STATUS { ALIVE, DEAD };

/**
 * Macro for defining agent transition functions with the correct input. Must always be a device function to be called by CUDA.
 *
 * struct SomeAgentFunction {
 *     __device__ __forceinline__ FLAME_GPU_AGENT_STATUS operator()(FLAMEGPU_DEVICE_API *FLAMEGPU) const {
 *         // do something
 *         return 0;
 *     }
 * };
 *}
 */
#define FLAMEGPU_AGENT_FUNCTION(funcName)\
struct funcName ## _impl {\
    __device__ __forceinline__ FLAME_GPU_AGENT_STATUS operator()(FLAMEGPU_DEVICE_API *FLAMEGPU) const;\
};\
funcName ## _impl funcName;\
__device__ __forceinline__ FLAME_GPU_AGENT_STATUS funcName ## _impl::operator()(FLAMEGPU_DEVICE_API *FLAMEGPU) const

// Advanced macro for defining agent transition functions
#define FLAMEGPU_AGENT_FUNC __device__ __forceinline__

/** @brief    A flame gpu api class for the device runtime only
 *
 * This class should only be used by the device and never created on the host. It is safe for each agent function to create a copy of this class on the device. Any singleton type
 * behaviour is handled by the curveInstance class. This will ensure that initialisation of the curve (C) library is done only once.
 */
class FLAMEGPU_DEVICE_API {
    // Friends have access to TID() & TS_ID()
    template<typename AgentFunction>
    friend __global__ void agent_function_wrapper(Curve::NamespaceHash, Curve::NamespaceHash, Curve::NamespaceHash, Curve::NamespaceHash, const int, const unsigned int, const unsigned int);

 public:
    /**
     * @param _thread_in_layer_offset This offset can be added to TID to give a thread-safe unique index for the thread
     */
    __device__ FLAMEGPU_DEVICE_API(const unsigned int &_thread_in_layer_offset, const Curve::NamespaceHash &modelname_hash)
        : random(AgentRandom(TID()+_thread_in_layer_offset)),
        environment(DeviceEnvironment(modelname_hash)),
        thread_in_layer_offset(_thread_in_layer_offset) { }

    template<typename T, unsigned int N> __device__
    T getVariable(const char(&variable_name)[N]);

    template<typename T, unsigned int N> __device__
    void setVariable(const char(&variable_name)[N], T value);

    template<typename T, unsigned int N> __device__
        T getMessageVariable(const char(&variable_name)[N]);

    template<typename T, unsigned int N> __device__
        void setMessageVariable(const char(&variable_name)[N], T value);

    template<typename T, unsigned int N> __device__
        void addMessage(const char(&variable_name)[N], T value);

    template<unsigned int N> __device__
        MessageList GetMessageIterator(const char(&message_name)[N]);

    /**
    * \brief
    * \param  messageList_Size
    */
    __device__ void setMessageListSize(unsigned int messageList_Size) {
        messageListSize = messageList_Size;
    }

    /**
     * \brief
     * \param agentname_hash
     */
    __device__ void setAgentNameSpace(Curve::NamespaceHash agentname_hash) {
        agent_func_name_hash = agentname_hash;
    }

    /**
     * \brief
     * \param messagename_hash
     */
    __device__ void setMessageInpNameSpace(Curve::NamespaceHash messagename_hash) {
        messagename_inp_hash = messagename_hash;
    }

    /**
     * \brief
     * \param messagename_hash
     */
    __device__ void setMessageOutpNameSpace(Curve::NamespaceHash messagename_hash) {
        messagename_outp_hash = messagename_hash;
    }

    /**
    * Provides access to random functionality inside agent functions
    * @note random state isn't stored within the object, so it can be const
    */
    const AgentRandom random;
    const DeviceEnvironment environment;

 private:
    Curve::NamespaceHash agent_func_name_hash;
    Curve::NamespaceHash messagename_inp_hash;
    Curve::NamespaceHash messagename_outp_hash;
    MessageList messageList;

    unsigned int  messageListSize;

    unsigned int thread_in_layer_offset;
    /**
     * Thread index
     */
    __forceinline__ __device__ static unsigned int TID() {
        /*
        // 3D version
        auto blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
        auto threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x)
            + threadIdx.x;
        return threadId;*/
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
        return blockIdx.x * blockDim.x +threadIdx.x;
    }
    /**
     * Thread-safe index
     */
    __forceinline__ __device__ unsigned int TS_ID() const {
        return thread_in_layer_offset + TID();
    }
};

/******************************************************************************************************* Implementation ********************************************************/

// An example of how the getVariable should work
// 1) Given the string name argument use runtime hashing to get a variable unsigned int (call function in RuntimeHashing.h)
// 2) Call (a new local) getHashIndex function to check the actual index in the hash table for the variable name. Once found we have a pointer to the vector of data for that agent variable
// 3) Using the CUDA thread and block index (threadIdx.x) return the specific agent variable value from the vector
// Useful existing code to look at is CUDAAgentStateList setAgentData function
// Note that this is using the hashing to get a specific pointer for a given variable name. This is exactly what we want to do in the FLAME GPU API class

/**
 * \brief Gets an agent memory value
 * \param variable_name Name of memory variable to retrieve
 */
template<typename T, unsigned int N>
__device__ T FLAMEGPU_DEVICE_API::getVariable(const char(&variable_name)[N]) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index =  (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = Curve::getVariable<T>(variable_name, agent_func_name_hash , index);

    // return the variable from curve
    return value;
}

/**
 * \brief Sets an agent memory value
 * \param variable_name Name of memory variable to set
 * \param value Value to set it to
 */
template<typename T, unsigned int N>
__device__ void FLAMEGPU_DEVICE_API::setVariable(const char(&variable_name)[N], T value) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    Curve::setVariable<T>(variable_name , agent_func_name_hash,  value, index);
}

/**
* \brief Gets the value of a message variable
* \param variable_name Name of memory variable to retrieve
* \todo check the hashing
*/
template<typename T, unsigned int N>
__device__ T FLAMEGPU_DEVICE_API::getMessageVariable(const char(&variable_name)[N]) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = Curve::getVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, index);

    // return the variable from curve
    return value;
}

/**
* \brief Sets the value of a message variable
* \param variable_name Name of memory variable to set
* \param value Value to set it to
* \todo check the hashing
*/
template<typename T, unsigned int N>
__device__ void FLAMEGPU_DEVICE_API::setMessageVariable(const char(&variable_name)[N], T value) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    Curve::setVariable<T>(variable_name, agent_func_name_hash + messagename_outp_hash, value, index);
}

/**
* \brief adds a message
* \param variable_name Name of message variable to set
* \param value Value to set it to
*/
template<typename T, unsigned int N>
__device__ void FLAMEGPU_DEVICE_API::addMessage(const char(&variable_name)[N], T value) {  // message name or variable name
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    Curve::setVariable<T>(variable_name, agent_func_name_hash + messagename_outp_hash, value, index);
}

/**
* \brief Returns a message iterator
* \param msg_name Name of message
* \todo not quite right
*/
template<unsigned int N>
__device__ MessageList FLAMEGPU_DEVICE_API::GetMessageIterator(const char(&message_name)[N]) {
    messageList.setAgentNameSpace(agent_func_name_hash);
    messageList.setMessageInpNameSpace(messagename_inp_hash);
    messageList.setMessageListSize(messageListSize);

    return messageList;
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_DEVICE_API_H_
