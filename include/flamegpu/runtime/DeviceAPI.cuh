#ifndef INCLUDE_FLAMEGPU_RUNTIME_DEVICEAPI_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_DEVICEAPI_CUH_


#include <cassert>
#include <cstdint>
#include <limits>

#ifndef __CUDACC_RTC__
#include "flamegpu/runtime/detail/curve/curve.cuh"
#include "flamegpu/runtime/messaging_device.h"
#else
#include "dynamic/curve_rtc_dynamic.h"
#endif  // !_RTC
#include "flamegpu/runtime/utility/AgentRandom.cuh"
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"
#include "flamegpu/runtime/AgentFunction.cuh"
#include "flamegpu/runtime/AgentFunctionCondition.cuh"
#include "flamegpu/defines.h"

#ifdef USE_GLM
#ifdef __CUDACC__
#pragma diag_suppress = esa_on_defaulted_function_ignored
#endif
#include <glm/glm.hpp>
#endif

namespace flamegpu {

/**
 * @brief  FLAMEGPU_API is a singleton class for the device runtime
 *
 * \todo longer description
 */
class ReadOnlyDeviceAPI {
    // Friends have access to TID() & TS_ID()
    template<typename AgentFunctionCondition>
    friend __global__ void agent_function_condition_wrapper(
#if !defined(SEATBELTS) || SEATBELTS
        exception::DeviceExceptionBuffer *error_buffer,
#endif
        detail::curve::Curve::NamespaceHash,
        detail::curve::Curve::NamespaceHash,
        const unsigned int,
        curandState *,
        unsigned int *);

 public:
    /**
     * @param instance_id_hash CURVE hash of the CUDASimulation's instance id
     * @param agentfuncname_hash CURVE hash of the agent function
     * @param d_rng Pointer to the device random state buffer to be used
     */
    __device__ ReadOnlyDeviceAPI(
        const detail::curve::Curve::NamespaceHash &instance_id_hash,
        const detail::curve::Curve::NamespaceHash &agentfuncname_hash,
        curandState *&d_rng)
        : random(AgentRandom(&d_rng[getThreadIndex()]))
        , environment(DeviceEnvironment(instance_id_hash))
        , agent_func_name_hash(agentfuncname_hash) { }
    /**
     * Returns the specified variable from the currently executing agent
     * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
     * @tparam T Type of the agent variable being accessed
     * @tparam N Length of variable name, this should always be implicit if passing a string literal
     * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N> __device__
    T getVariable(const char(&variable_name)[N]) const;
    /**
     * Returns the specified variable array element from the currently executing agent
     * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
     * @param index Index of the element within the variable array to return
     * @tparam T Type of the agent variable being accessed
     * @tparam N The length of the array variable, as set within the model description hierarchy
     * @tparam M Length of variable_name, this should always be implicit if passing a string literal
     * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N, unsigned int M> __device__
    T getVariable(const char(&variable_name)[M], const unsigned int &index) const;
    /**
     * Returns the agent's unique identifier
     */
    __device__ id_t getID() {
        return getVariable<id_t>("_id");
    }

    /**
     * Access the current stepCount
     * @return the current step count, 0 indexed unsigned.
     */
    __forceinline__ __device__ unsigned int getStepCounter() const {
        return environment.getProperty<unsigned int>("_stepCount");
    }

    /**
     * Provides access to random functionality inside agent functions
     * @note random state isn't stored within the object, so it can be const
     */
    const AgentRandom random;
    /**
     * Provides access to environment variables inside agent functions
     */
    const ReadOnlyDeviceEnvironment environment;

    /**
     * Returns the current CUDA thread of the agent
     * All agents execute in a unique thread, but their associated thread may change between agent functions
     * Thread indices begin at 0 and continue to 1 below the number of agents executing
     */
    __forceinline__ __device__ static unsigned int getThreadIndex() {
        /*
        // 3D version
        auto blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
        auto threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
        return threadId;*/
#ifdef SEATBELTS
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
#endif
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

 protected:
    detail::curve::Curve::NamespaceHash agent_func_name_hash;
};

/** @brief    A flame gpu api class for the device runtime only
 *
 * This class provides access to model variables/state inside agent functions
 *
 * This class should only be used by the device and never created on the host. It is safe for each agent function to create a copy of this class on the device. Any singleton type
 * behaviour is handled by the curveInstance class. This will ensure that initialisation of the curve (C) library is done only once.
 * @tparam MessageIn Input message type (the form found in flamegpu/runtime/messaging.h, MessageNone etc)
 * @tparam MessageOut Output message type (the form found in flamegpu/runtime/messaging.h, MessageNone etc)
 */
template<typename MessageIn, typename MessageOut>
class DeviceAPI {
    // Friends have access to TID() & TS_ID()
    template<typename AgentFunction, typename _MessageIn, typename _MessageOut>
    friend __global__ void agent_function_wrapper(
#if !defined(SEATBELTS) || SEATBELTS
        exception::DeviceExceptionBuffer *error_buffer,
#endif
        detail::curve::Curve::NamespaceHash,
        detail::curve::Curve::NamespaceHash,
        detail::curve::Curve::NamespaceHash,
        detail::curve::Curve::NamespaceHash,
        detail::curve::Curve::NamespaceHash,
        id_t*,
        const unsigned int,
        const void *,
        const void *,
        curandState *,
        unsigned int *,
        unsigned int *,
        unsigned int *);

 public:
    /**
     * Collection of DeviceAPI functions related to agent birth
     */
    class AgentOut {
     public:
        /**
         * Constructor
         * @param aoh Agent output hash (as required for accessing Curve)
         * @param d_agent_output_nextID Pointer to global memory holding the IDs to be assigned to new agents (selected via atomic inc)
         * @param scan_flag_agentOutput Pointer to (the start of) buffer of scan flags to be set true if this thread outputs an agent
         */
        __device__ AgentOut(const detail::curve::Curve::NamespaceHash &aoh, id_t *&d_agent_output_nextID, unsigned int *&scan_flag_agentOutput)
            : agent_output_hash(aoh)
            , scan_flag(scan_flag_agentOutput)
            , nextID(d_agent_output_nextID) { }
        /**
         * Sets a variable in a new agent to be output after the agent function has completed
         * @param variable_name The name of the variable
         * @param value The value to set the variable
         * @tparam T The type of the variable, as set within the model description hierarchy
         * @tparam N Variable name length, this should be ignored as it is implicitly set
         * @note Any agent variables not set will remain as their default values
         * @note Calling AgentOut::setVariable() or AgentOut::getID() will trigger agent output
         * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         */
        template<typename T, unsigned int N>
        __device__ void setVariable(const char(&variable_name)[N], T value) const;
        /**
         * Sets an element of an array variable in a new agent to be output after the agent function has completed
         * @param variable_name The name of the array variable
         * @param index The index to set within the array variable
         * @param value The value to set the element of the array element
         * @tparam T The type of the variable, as set within the model description hierarchy
         * @tparam N The length of the array variable, as set within the model description hierarchy
         * @tparam M Variable name length, this should be ignored as it is implicitly set
         * @note Any agent variables not set will remain as their default values
         * @note Calling AgentOut::setVariable() or AgentOut::getID() will trigger agent output
         * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
         * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
         */
        template<typename T, unsigned int N, unsigned int M>
        __device__ void setVariable(const char(&variable_name)[M], const unsigned int &index, T value) const;
        /**
         * Return the ID of the agent to be created
         * @note Calling AgentOut::setVariable() or AgentOut::getID() will trigger agent output
         */
        __device__ id_t getID() const;

     private:
        /**
         * Sets scan flag and id
         */
        __device__ void genID() const;
        /**
         * Curve hash used for accessing new agent variables
         */
        const detail::curve::Curve::NamespaceHash agent_output_hash;
        /**
         * Scan flag, defaults to 0, set to 1, to mark than agent is output
         */
        unsigned int* const scan_flag;
        /**
         * Agent id if set
         * @note mutable, because this object is always const
         */
        mutable id_t id = ID_NOT_SET;
        /**
         * Ptr to global address storing a counter to track the next available agent ID for the agent type being output
         */
        id_t *nextID;
    };
    /**
     * Constructs the device-only API class instance.
     * @param instance_id_hash CURVE hash of the CUDASimulation's instance id
     * @param agentfuncname_hash Combined CURVE hashes of agent name and func name
     * @param _agent_output_hash Combined CURVE hashes for agent output
     * @param d_agent_output_nextID If agent birth is enabled, a pointer to the next available ID in global memory. Device agent birth will atomically increment this value to allocate IDs.
     * @param d_rng Device pointer to curand state for this kernel, index 0 should for TID()==0
     * @param scanFlag_agentOutput Array for agent output scan flag
     * @param message_in Input message handler
     * @param message_out Output message handler
     */
    __device__ DeviceAPI(
        const detail::curve::Curve::NamespaceHash &instance_id_hash,
        const detail::curve::Curve::NamespaceHash &agentfuncname_hash,
        const detail::curve::Curve::NamespaceHash &_agent_output_hash,
        id_t *&d_agent_output_nextID,
        curandState *&d_rng,
        unsigned int *&scanFlag_agentOutput,
        typename MessageIn::In &&message_in,
        typename MessageOut::Out &&message_out)
        : message_in(message_in)
        , message_out(message_out)
        , agent_out(AgentOut(_agent_output_hash, d_agent_output_nextID, scanFlag_agentOutput))
        , random(AgentRandom(&d_rng[getThreadIndex()]))
        , environment(DeviceEnvironment(instance_id_hash))
        , agent_func_name_hash(agentfuncname_hash)
    { }
        /**
     * Returns the specified variable from the currently executing agent
     * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
     * @tparam T Type of the agent variable being accessed
     * @tparam N Length of variable name, this should always be implicit if passing a string literal
     * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N> __device__
    T getVariable(const char(&variable_name)[N]) const;
    /**
     * Returns the specified variable array element from the currently executing agent
     * @param variable_name name used for accessing the variable, this value should be a string literal e.g. "foobar"
     * @param index Index of the element within the variable array to return
     * @tparam T Type of the agent variable being accessed
     * @tparam N The length of the array variable, as set within the model description hierarchy
     * @tparam M Length of variable_name, this should always be implicit if passing a string literal
     * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N, unsigned int M> __device__
    T getVariable(const char(&variable_name)[M], const unsigned int &index) const;
    /**
     * Sets a variable within the currently executing agent
     * @param variable_name The name of the variable
     * @param value The value to set the variable
     * @tparam T The type of the variable, as set within the model description hierarchy
     * @tparam N variable_name length, this should be ignored as it is implicitly set
     * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N>
    __device__ void setVariable(const char(&variable_name)[N], T value);
    /**
     * Sets an element of an array variable within the currently executing agent
     * @param variable_name The name of the array variable
     * @param index The index to set within the array variable
     * @param value The value to set the element of the array element
     * @tparam T The type of the variable, as set within the model description hierarchy
     * @tparam N The length of the array variable, as set within the model description hierarchy
     * @tparam M variable_name length, this should be ignored as it is implicitly set
     * @throws exception::DeviceError If name is not a valid variable within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If T is not the type of variable 'name' within the agent (flamegpu must be built with SEATBELTS enabled for device error checking)
     * @throws exception::DeviceError If index is out of bounds for the variable array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
     */
    template<typename T, unsigned int N, unsigned int M>
    __device__ void setVariable(const char(&variable_name)[M], const unsigned int &index, const T &value);
    /**
     * Returns the agent's unique identifier
     */
    __device__ id_t getID() {
        return getVariable<id_t>("_id");
    }

    /**
     * Access the current stepCount
     * @return the current step count, 0 indexed unsigned.
     */
    __forceinline__ __device__ unsigned int getStepCounter() const {
        return environment.getProperty<unsigned int>("_stepCount");
    }

    /**
     * Returns the current CUDA thread of the agent
     * All agents execute in a unique thread, but their associated thread may change between agent functions
     * Thread indices begin at 0 and continue to 1 below the number of agents executing
     */
    __forceinline__ __device__ static unsigned int getThreadIndex() {
        /*
        // 3D version
        auto blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
        auto threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
        return threadId;*/
#if !defined(SEATBELTS) || SEATBELTS
        assert(blockDim.y == 1);
        assert(blockDim.z == 1);
        assert(gridDim.y == 1);
        assert(gridDim.z == 1);
#endif
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    /**
     * Provides access to message read functionality inside agent functions
     */
    const typename MessageIn::In message_in;
    /**
     * Provides access to message write functionality inside agent functions
     */
    const typename MessageOut::Out message_out;
    /**
     * Provides access to agent output functionality inside agent functions
     */
    const AgentOut agent_out;
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
     detail::curve::Curve::NamespaceHash agent_func_name_hash;
};


/******************************************************************************************************* Implementation ********************************************************/

template<typename T, unsigned int N>
__device__ T ReadOnlyDeviceAPI::getVariable(const char(&variable_name)[N]) const {
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = detail::curve::Curve::getAgentVariable<T>(variable_name, agent_func_name_hash , index);

    // return the variable from curve
    return value;
}
template<typename T, unsigned int N, unsigned int M>
__device__ T ReadOnlyDeviceAPI::getVariable(const char(&variable_name)[M], const unsigned int &array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = detail::curve::Curve::getAgentArrayVariable<T, N>(variable_name, agent_func_name_hash , index, array_index);

    // return the variable from curve
    return value;
}

template<typename MessageIn, typename MessageOut>
template<typename T, unsigned int N>
__device__ T DeviceAPI<MessageIn, MessageOut>::getVariable(const char(&variable_name)[N]) const {
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = detail::curve::Curve::getAgentVariable<T>(variable_name, agent_func_name_hash , index);

    // return the variable from curve
    return value;
}

template<typename MessageIn, typename MessageOut>
template<typename T, unsigned int N, unsigned int M>
__device__ T DeviceAPI<MessageIn, MessageOut>::getVariable(const char(&variable_name)[M], const unsigned int &array_index) const {
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // get the value from curve
    T value = detail::curve::Curve::getAgentArrayVariable<T, N>(variable_name, agent_func_name_hash , index, array_index);

    // return the variable from curve
    return value;
}

template<typename MessageIn, typename MessageOut>
template<typename T, unsigned int N>
__device__ void DeviceAPI<MessageIn, MessageOut>::setVariable(const char(&variable_name)[N], T value) {
    if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in DeviceAPI::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    // set the variable using curve
    detail::curve::Curve::setAgentVariable<T>(variable_name, agent_func_name_hash, value, index);
}
template<typename MessageIn, typename MessageOut>
template<typename T, unsigned int N, unsigned int M>
__device__ void DeviceAPI<MessageIn, MessageOut>::setVariable(const char(&variable_name)[M], const unsigned int &array_index, const T &value) {
    if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
        DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in DeviceAPI::setVariable().\n", variable_name);
#endif
        return;  // Fail silently
    }
    // simple indexing assumes index is the thread number (this may change later)
    const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    // set the variable using curve
    detail::curve::Curve::setAgentArrayVariable<T, N>(variable_name , agent_func_name_hash,  value, index, array_index);
}

template<typename MessageIn, typename MessageOut>
template<typename T, unsigned int N>
__device__ void DeviceAPI<MessageIn, MessageOut>::AgentOut::setVariable(const char(&variable_name)[N], T value) const {
    if (agent_output_hash) {
        if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
            DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in AgentOut::setVariable().\n", variable_name);
#endif
            return;  // Fail silently
        }
        if (agent_output_hash) {
            // simple indexing assumes index is the thread number (this may change later)
            const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

            // set the variable using curve
            detail::curve::Curve::setNewAgentVariable<T>(variable_name, agent_output_hash, value, index);

            // Mark scan flag
            genID();
        }
#if !defined(SEATBELTS) || SEATBELTS
    } else {
        DTHROW("Agent output must be enabled per agent function when defining the model.\n");
#endif
    }
}
template<typename MessageIn, typename MessageOut>
template<typename T, unsigned int N, unsigned int M>
__device__ void DeviceAPI<MessageIn, MessageOut>::AgentOut::setVariable(const char(&variable_name)[M], const unsigned int &array_index, T value) const {
    if (agent_output_hash) {
        if (variable_name[0] == '_') {
#if !defined(SEATBELTS) || SEATBELTS
            DTHROW("Variable names starting with '_' are reserved for internal use, with '%s', in AgentOut::setVariable().\n", variable_name);
#endif
            return;  // Fail silently
        }
        // simple indexing assumes index is the thread number (this may change later)
        const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

        // set the variable using curve
        detail::curve::Curve::setNewAgentArrayVariable<T, N>(variable_name, agent_output_hash, value, index, array_index);

        // Mark scan flag
        genID();
#if !defined(SEATBELTS) || SEATBELTS
    } else {
        DTHROW("Agent output must be enabled per agent function when defining the model.\n");
#endif
    }
}

template<typename MessageIn, typename MessageOut>
__device__ id_t DeviceAPI<MessageIn, MessageOut>::AgentOut::getID() const {
    if (agent_output_hash) {
        genID();
        return this->id;
    }
#if !defined(SEATBELTS) || SEATBELTS
    DTHROW("Agent output must be enabled per agent function when defining the model.\n");
#endif
    return ID_NOT_SET;
}
#ifdef __CUDACC__
template<typename MessageIn, typename MessageOut>
__device__ void DeviceAPI<MessageIn, MessageOut>::AgentOut::genID() const {
    // Only assign id and scan flag once
    if (this->id == ID_NOT_SET) {
        this->id = atomicInc(this->nextID, std::numeric_limits<id_t>().max());
        const unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;
        detail::curve::Curve::setNewAgentVariable<id_t>("_id", agent_output_hash, this->id, index);  // Can't use ID_VARIABLE_NAME inline, as it isn't of char[N] type
        this->scan_flag[index] = 1;
    }
}
#endif

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DEVICEAPI_CUH_
