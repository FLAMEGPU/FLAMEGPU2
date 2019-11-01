 /**
 * @file CUDAAgentStateList.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAAgentStateList.h"

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/pop/AgentStateMemory.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"

/**
* CUDAAgentStateList class
* @brief populates CUDA agent map
*/
CUDAAgentStateList::CUDAAgentStateList(CUDAAgent& cuda_agent) : agent(cuda_agent) {
    // allocate state lists
    allocateDeviceAgentList(d_list);
    allocateDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().requiresAgentCreation())
        allocateDeviceAgentList(d_new_list);
}

/**
 * A destructor.
 * @brief Destroys the CUDAAgentStateList object
 */
CUDAAgentStateList::~CUDAAgentStateList() {
}

void CUDAAgentStateList::cleanupAllocatedData() {
    // clean up
    releaseDeviceAgentList(d_list);
    releaseDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().requiresAgentCreation()) {
        releaseDeviceAgentList(d_new_list);
    }
}

/**
* @brief Allocates Device agent list
* @param variable of type CUDAMemoryMap type
* @return none
*/
void CUDAAgentStateList::allocateDeviceAgentList(CUDAMemoryMap &memory_map) {
    // we use the agents memory map to iterate the agent variables and do allocation within our GPU hash map
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

    // for each variable allocate a device array and add to map
    for (const MemoryMapPair& mm : mem) {
        // get the variable name
        std::string var_name = mm.first;

        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(mm.first);

        // do the device allocation
        void * d_ptr;

#ifdef UNIFIED_GPU_MEMORY
        // unified memory allocation
        gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&d_ptr), var_size * agent.getMaximumListSize()))
#else
        // non unified memory allocation
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_ptr), var_size * agent.getMaximumListSize()));
#endif

        // store the pointer in the map
        memory_map.insert(CUDAMemoryMap::value_type(var_name, d_ptr));
    }
}

/**
* @brief Frees
* @param variable of type CUDAMemoryMap struct type
* @return none
*/
void CUDAAgentStateList::releaseDeviceAgentList(CUDAMemoryMap& memory_map) {
    // for each device pointer in the cuda memory map we need to free these
    for (const CUDAMemoryMapPair& mm : memory_map) {
        // free the memory on the device
        gpuErrchk(cudaFree(mm.second));
    }
}

/**
* @brief
* @param variable of type CUDAMemoryMap struct type
* @return none
*/
void CUDAAgentStateList::zeroDeviceAgentList(CUDAMemoryMap& memory_map) {
    // for each device pointer in the cuda memory map set the values to 0
    for (const CUDAMemoryMapPair& mm : memory_map) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(mm.first);

        // set the memory to zero
        gpuErrchk(cudaMemset(mm.second, 0, var_size*agent.getMaximumListSize()));
    }
}

/**
* @brief
* @param AgenstStateMemory object
* @return none
* @todo
*/
void CUDAAgentStateList::setAgentData(const AgentStateMemory &state_memory) {
    // check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription())) {
        // throw std::runtime_error("CUDA Agent uses different agent description.");
        throw InvalidCudaAgentDesc();
    }

    // copy raw agent data to device pointers
    for (CUDAMemoryMapPair m : d_list) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(m.first);

        // get the vector
        const GenericMemoryVector &m_vec = state_memory.getReadOnlyMemoryVector(m.first);

        // get pointer to vector data
        const void * v_data = m_vec.getReadOnlyDataPtr();

        // set the current list size
        current_list_size = state_memory.getStateListSize();

        // copy the host data to the GPU
        gpuErrchk(cudaMemcpy(m.second, v_data, var_size*current_list_size, cudaMemcpyHostToDevice));
    }
}

void CUDAAgentStateList::getAgentData(AgentStateMemory &state_memory) {
    // check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription())) {
        // throw std::runtime_error("CUDA Agent uses different agent description.");
        throw InvalidCudaAgentDesc();
    }

    // copy raw agent data to device pointers
    for (CUDAMemoryMapPair m : d_list) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(m.first);

        // get the vector
        GenericMemoryVector &m_vec = state_memory.getMemoryVector(m.first);

        // get pointer to vector data
        void * v_data = m_vec.getDataPtr();

        // check  the current list size
        if (current_list_size > state_memory.getPopulationCapacity())
            throw InvalidMemoryCapacity("Current GPU state list size exceed the state memory available!");

        // copy the GPU data to host
        gpuErrchk(cudaMemcpy(v_data, m.second, var_size*current_list_size, cudaMemcpyDeviceToHost));

        // set the new state list size
        state_memory.overrideStateListSize(current_list_size);
    }
}

void* CUDAAgentStateList::getAgentListVariablePointer(std::string variable_name) {
    CUDAMemoryMap::iterator mm = d_list.find(variable_name);
    if (mm == d_list.end()) {
        // TODO: Error variable not found in agent state list
        return 0;
    }

    return mm->second;
}

void CUDAAgentStateList::zeroAgentData() {
    zeroDeviceAgentList(d_list);
    zeroDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().requiresAgentCreation())
        zeroDeviceAgentList(d_new_list);
}

// the actual number of agents in this state
unsigned int CUDAAgentStateList::getCUDAStateListSize() const {
    return current_list_size;
}
