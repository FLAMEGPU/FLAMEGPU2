#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAME_FUNCTIONS_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAME_FUNCTIONS_API_H_

/**
 * @file flame_functions_api.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief  FLAMEGPU_API is a singleton class for the device runtime
 *
 * \todo longer description
 */

#include "flamegpu/gpu/CUDAErrorChecking.h"            // required for CUDA error handling functions
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/runtime/messagelist.h"

// TODO: Some example code of the handle class and an example function
// ! FLAMEGPU_API is a singleton class
class FLAMEGPU_API;  // Forward declaration (class defined below)

// ! FLAMEGPU function return type
enum FLAME_GPU_AGENT_STATUS { ALIVE, DEAD };

/**
 * @brief FLAMEGPU function pointer definition
 */
typedef FLAME_GPU_AGENT_STATUS(*FLAMEGPU_AGENT_FUNCTION_POINTER)(FLAMEGPU_API *api);

/**
 * Macro for defining agent transition functions with the correct input. Must always be a device function to be called by CUDA.
 *
 * FLAMEGPU_AGENT_FUNCTION(move_func) {
 *   int x = FLAMEGPU.getVariable<int>("x");
 *   FLAMEGPU.setVariable<int>("x", x*10);
 * return ALIVE;
 *}
 */
// #define FLAMEGPU_AGENT_FUNCTION(funcName) __device__ FLAME_GPU_AGENT_STATUS funcName(FLAMEGPU_API* FLAMEGPU)

#define FLAMEGPU_AGENT_FUNCTION(funcName) \
__device__ FLAME_GPU_AGENT_STATUS funcName ## _impl(FLAMEGPU_API* FLAMEGPU); \
__device__ FLAMEGPU_AGENT_FUNCTION_POINTER funcName = funcName ## _impl;\
__device__ FLAME_GPU_AGENT_STATUS funcName ## _impl(FLAMEGPU_API* FLAMEGPU)

/** @brief    A flame gpu api class for the device runtime only
 *
 * This class should only be used by the device and never created on the host. It is safe for each agent function to create a copy of this class on the device. Any singleton type
 * behaviour is handled by the curveInstance class. This will ensure that initialisation of the curve (C) library is done only once.
 */
class FLAMEGPU_API {
 public:
    __device__ FLAMEGPU_API() {}

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
    __device__ void setAgentNameSpace(CurveNamespaceHash agentname_hash) {
agent_func_name_hash = agentname_hash;}

    /**
     * \brief
     * \param messagename_hash
     */
    __device__ void setMessageInpNameSpace(CurveNamespaceHash messagename_hash) {
        messagename_inp_hash = messagename_hash;
    }

    /**
     * \brief
     * \param messagename_hash
     */
    __device__ void setMessageOutpNameSpace(CurveNamespaceHash messagename_hash) {
        messagename_outp_hash = messagename_hash;
    }

 private:
    CurveNamespaceHash agent_func_name_hash;
    CurveNamespaceHash messagename_inp_hash;
    CurveNamespaceHash messagename_outp_hash;
    MessageList messageList;

    unsigned int  messageListSize;
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
__device__ T FLAMEGPU_API::getVariable(const char(&variable_name)[N]) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index =  (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = curveGetVariable<T>(variable_name, agent_func_name_hash , index);

    // return the variable from curve
    return value;
}

/**
 * \brief Sets an agent memory value
 * \param variable_name Name of memory variable to set
 * \param value Value to set it to
 */
template<typename T, unsigned int N>
__device__ void FLAMEGPU_API::setVariable(const char(&variable_name)[N], T value) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    curveSetVariable<T>(variable_name , agent_func_name_hash,  value, index);
}

/**
* \brief Gets the value of a message variable
* \param variable_name Name of memory variable to retrieve
* \todo check the hashing
*/
template<typename T, unsigned int N>
__device__ T FLAMEGPU_API::getMessageVariable(const char(&variable_name)[N]) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = curveGetVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, index);

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
__device__ void FLAMEGPU_API::setMessageVariable(const char(&variable_name)[N], T value) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    curveSetVariable<T>(variable_name, agent_func_name_hash + messagename_outp_hash, value, index);
}

/**
* \brief adds a message
* \param variable_name Name of message variable to set
* \param value Value to set it to
*/
template<typename T, unsigned int N>
__device__ void FLAMEGPU_API::addMessage(const char(&variable_name)[N], T value) {// message name or variable name
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    curveSetVariable<T>(variable_name, agent_func_name_hash + messagename_outp_hash, value, index);
}

/**
* \brief Returns a message iterator
* \param msg_name Name of message
* \todo not quite right
*/
template<unsigned int N>
__device__ MessageList FLAMEGPU_API::GetMessageIterator(const char(&message_name)[N]) {
    messageList.setAgentNameSpace(agent_func_name_hash);
    messageList.setMessageInpNameSpace(messagename_inp_hash);
    messageList.setMessageListSize(messageListSize);

    return messageList;
}

#endif // INCLUDE_FLAMEGPU_RUNTIME_FLAME_FUNCTIONS_API_H_
