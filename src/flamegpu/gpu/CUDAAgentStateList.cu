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

namespace flamegpu_internal {
    extern __device__ unsigned int *ds_agent_scan_flag;
    extern __device__ unsigned int *ds_agent_position;
}  // namespace flamegpu_internal

/**
* CUDAAgentStateList class
* @brief populates CUDA agent map
*/
CUDAAgentStateList::CUDAAgentStateList(CUDAAgent& cuda_agent)
    : current_list_size(0)
    , agent(cuda_agent) {
    // allocate state lists
    allocateDeviceAgentList(d_list);
    allocateDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().isOutputOnDevice())
        allocateDeviceAgentList(d_new_list);
}

/**
 * A destructor.
 * @brief Destroys the CUDAAgentStateList object
 */
CUDAAgentStateList::~CUDAAgentStateList() {
    cleanupAllocatedData();
}

void CUDAAgentStateList::cleanupAllocatedData() {
    // clean up
    releaseDeviceAgentList(d_list);
    releaseDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().isOutputOnDevice()) {
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
    const auto &mem = agent.getAgentDescription().variables;

    // for each variable allocate a device array and add to map
    for (const auto &mm : mem) {
        // get the variable name
        std::string var_name = mm.first;

        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(mm.first).type_size;

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
        size_t var_size = agent.getAgentDescription().variables.at(mm.first).type_size;

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
        THROW InvalidCudaAgentDesc("Agent State memory has different description to CUDA Agent ('%s'), "
            "in CUDAAgentStateList::setAgentData().",
            agent.getAgentDescription().name.c_str());
    }

    // copy raw agent data to device pointers
    for (CUDAMemoryMapPair m : d_list) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(m.first).type_size;

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
        THROW InvalidCudaAgentDesc("Agent State memory has different description to CUDA Agent ('%s'), "
            "in CUDAAgentStateList::getAgentData().",
            agent.getAgentDescription().name.c_str());
    }

    // copy raw agent data to device pointers
    for (CUDAMemoryMapPair m : d_list) {
        // get the variable size from agent description
        size_t var_size = agent.getAgentDescription().variables.at(m.first).type_size;

        // get the vector
        GenericMemoryVector &m_vec = state_memory.getMemoryVector(m.first);

        // get pointer to vector data
        void * v_data = m_vec.getDataPtr();

        // check  the current list size
        if (current_list_size > state_memory.getPopulationCapacity()) {
            THROW InvalidMemoryCapacity("Current GPU state list size (%u) exceeds the state memory available (%u), "
                "in CUDAAgentStateList::getAgentData()",
                current_list_size, state_memory.getPopulationCapacity());
        }
        // copy the GPU data to host
        gpuErrchk(cudaMemcpy(v_data, m.second, var_size*current_list_size, cudaMemcpyDeviceToHost));

        // set the new state list size
        state_memory.overrideStateListSize(current_list_size);
    }
}

void* CUDAAgentStateList::getAgentListVariablePointer(std::string variable_name) const {
    CUDAMemoryMap::const_iterator mm = d_list.find(variable_name);
    if (mm == d_list.end()) {
        // TODO: Error variable not found in agent state list
        return 0;
    }

    return mm->second;
}

void CUDAAgentStateList::zeroAgentData() {
    zeroDeviceAgentList(d_list);
    zeroDeviceAgentList(d_swap_list);
    if (agent.getAgentDescription().isOutputOnDevice())
        zeroDeviceAgentList(d_new_list);
}

// the actual number of agents in this state
unsigned int CUDAAgentStateList::getCUDAStateListSize() const {
    return current_list_size;
}

__global__ void scatter_living_agents(
    size_t typeLen,
    char * const __restrict__ in,
    char * out,
    const unsigned int streamId) {
    // global thread index
    int index = (blockIdx.x*blockDim.x) + threadIdx.x;

    // if optional message is to be written
    if (flamegpu_internal::ds_agent_scan_flag[index] == 1) {
        int output_index = flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].position[index];
        memcpy(out + (output_index * typeLen), in + (index * typeLen), typeLen);
    }
}
void CUDAAgentStateList::scatter(const unsigned int &streamId) {
    int blockSize = 0;  // The launch configurator returned block size
    int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
    int gridSize = 0;  // The actual grid size needed, based on input size

                       // calculate the grid block size for main agent function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scatter_living_agents, 0, getCUDAStateListSize());
    //! Round up according to CUDAAgent state list size
    gridSize = (getCUDAStateListSize() + blockSize - 1) / blockSize;
    // for each variable, scatter from swap to regular
    for (const auto &v : agent.getAgentDescription().variables) {
        char *in_p = reinterpret_cast<char*>(d_swap_list.at(v.first));
        char *out_p = reinterpret_cast<char*>(d_list.at(v.first));

        scatter_living_agents << <gridSize, blockSize >> > (v.second.type_size, in_p, out_p, streamId);
    }
    gpuErrchkLaunch();
}
