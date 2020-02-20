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
#include "flamegpu/runtime/utility/AgentRandom.cuh"
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"
#include "flamegpu/gpu/CUDAScanCompaction.h"

class FLAMEGPU_READ_ONLY_DEVICE_API {
    // Friends have access to TID() & TS_ID()
    template<typename AgentFunctionCondition>
    friend __global__ void agent_function_condition_wrapper(
        Curve::NamespaceHash,
        Curve::NamespaceHash,
        const int,
        const unsigned int,
        const unsigned int);

 public:
    /**
     * @param _thread_in_layer_offset This offset can be added to TID to give a thread-safe unique index for the thread
     * @param modelname_hash CURVE hash of the model's name
     * @param _streamId Index used for accessing global members in a stream safe manner
    */
    __device__ FLAMEGPU_READ_ONLY_DEVICE_API(const unsigned int &_thread_in_layer_offset, const Curve::NamespaceHash &modelname_hash, const Curve::NamespaceHash &agentfuncname_hash, const unsigned int &_streamId)
        : random(AgentRandom(TID() + _thread_in_layer_offset))
        , environment(DeviceEnvironment(modelname_hash))
        , agent_func_name_hash(agentfuncname_hash)
        , thread_in_layer_offset(_thread_in_layer_offset)
        , streamId(_streamId)
    { }

    template<typename T, unsigned int N> __device__
        T getVariable(const char(&variable_name)[N]);

    /**
     * Provides access to random functionality inside agent functions
     * @note random state isn't stored within the object, so it can be const
     */
    const AgentRandom random;
    /**
     * Provides access to environment variables inside agent functions
     */
    const DeviceEnvironment environment;

 protected:
    Curve::NamespaceHash agent_func_name_hash;

    unsigned int thread_in_layer_offset;
    unsigned int streamId;
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
        return blockIdx.x * blockDim.x + threadIdx.x;
    }
    /**
    * Thread-safe index
    */
    __forceinline__ __device__ unsigned int TS_ID() const {
        return thread_in_layer_offset + TID();
    }
};

/** @brief    A flame gpu api class for the device runtime only
 *
 * This class provides access to model variables/state inside agent functions
 *
 * This class should only be used by the device and never created on the host. It is safe for each agent function to create a copy of this class on the device. Any singleton type
 * behaviour is handled by the curveInstance class. This will ensure that initialisation of the curve (C) library is done only once.
 * @tparam MsgIn Input message type (the form found in flamegpu/runtime/messaging.h, MsgNone etc)
 * @tparam MsgOut Output message type (the form found in flamegpu/runtime/messaging.h, MsgNone etc)
 */
template<typename MsgIn, typename MsgOut>
class FLAMEGPU_DEVICE_API : public FLAMEGPU_READ_ONLY_DEVICE_API{
    // Friends have access to TID() & TS_ID()
    template<typename AgentFunction, typename _MsgIn, typename _MsgOut>
    friend __global__ void agent_function_wrapper(
        Curve::NamespaceHash,
        Curve::NamespaceHash,
        Curve::NamespaceHash,
        Curve::NamespaceHash,
        const int,
        const void *messagelist_metadata,
        const unsigned int,
        const unsigned int);

 public:
    /**
     * @param _thread_in_layer_offset This offset can be added to TID to give a thread-safe unique index for the thread
     * @param modelname_hash CURVE hash of the model's name
     * @param agentfuncname_hash Combined CURVE hashes of agent name and func name
     * @param _streamId Index used for accessing global members in a stream safe manner
     * @param msg_in Input message handler
     * @param msg_out Output message handler
     */
    __device__ FLAMEGPU_DEVICE_API(
        const unsigned int &_thread_in_layer_offset,
        const Curve::NamespaceHash &modelname_hash,
        const Curve::NamespaceHash &agentfuncname_hash,
        const unsigned int &_streamId,
        typename MsgIn::In &&msg_in,
        typename MsgOut::Out &&msg_out)
        : FLAMEGPU_READ_ONLY_DEVICE_API(_thread_in_layer_offset, modelname_hash, agentfuncname_hash, _streamId)
        , message_in(msg_in)
        , message_out(msg_out)
    { }

    template<typename T, unsigned int N> __device__
    void setVariable(const char(&variable_name)[N], T value);

    /**
     * Provides access to message read functionality inside agent functions
     */
    const typename MsgIn::In message_in;
    /**
     * Provides access to message write functionality inside agent functions
     */
    const typename MsgOut::Out message_out;
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
__device__ T FLAMEGPU_READ_ONLY_DEVICE_API::getVariable(const char(&variable_name)[N]) {
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
template<typename MsgIn, typename MsgOut>
template<typename T, unsigned int N>
__device__ void FLAMEGPU_DEVICE_API<MsgIn, MsgOut>::setVariable(const char(&variable_name)[N], T value) {
    // simple indexing assumes index is the thread number (this may change later)
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    Curve::setVariable<T>(variable_name , agent_func_name_hash,  value, index);
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_DEVICE_API_H_
