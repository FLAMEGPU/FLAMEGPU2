#include "flamegpu/gpu/CUDASubAgentStateList.h"
#include "flamegpu/gpu/CUDASubAgent.h"
#include "flamegpu/model/SubModelData.h"

CUDASubAgentStateList::CUDASubAgentStateList(CUDASubAgent &cuda_agent, const std::shared_ptr<CUDAAgentStateList> &_master_list, const std::shared_ptr<SubAgentData> &_mapping)
    : CUDAAgentStateList(static_cast<CUDAAgent&>(cuda_agent))
    , master_list(_master_list)
    , mapping(_mapping) {
    // Bind to master list
    master_list->setDependentList(this, _mapping);
    // Replace mapped state lists
    if (agent.getMaximumListSize()) {
        // Unable to call virtual functions in a constructor
        // Therefore we reimplement allocateDeviceAgentList()
        CUDASubAgentStateList::allocateDeviceAgentList(d_list, agent.getMaximumListSize());
        CUDASubAgentStateList::allocateDeviceAgentList(d_swap_list, agent.getMaximumListSize());
        // Init condition state lists (for none mapped vars only)
        for (const auto &c : d_list) {
            if (mapping->variables.find(c.first) == mapping->variables.end())
                condition_d_list.emplace(c);
        }
        for (const auto &c : d_swap_list) {
            if (mapping->variables.find(c.first) == mapping->variables.end())
                condition_d_swap_list.emplace(c);
        }
        // Simple check to ensure init has completed properly
        assert(d_list.size() == agent.getAgentDescription().variables.size());
        assert(d_swap_list.size() == agent.getAgentDescription().variables.size());
        assert(condition_d_list.size() == agent.getAgentDescription().variables.size());
        assert(condition_d_swap_list.size() == agent.getAgentDescription().variables.size());
    }
}

void CUDASubAgentStateList::allocateDeviceAgentList(CUDAMemoryMap &memory_map, const unsigned int &allocSize) {
    // for each variable allocate a device array and add to map
    for (const auto &mm : agent.getAgentDescription().variables) {
        // get the variable name
        std::string sub_var_name = mm.first;
        auto map = mapping->variables.find(sub_var_name);
        // Only handle variables which are not mapped
        if (map == mapping->variables.end()) {
            // get the variable size from agent description
            const auto &var = agent.getAgentDescription().variables.at(sub_var_name);
            const size_t var_size = var.type_size;
            const unsigned int  var_elements = var.elements;

            // do the device allocation
            void * d_ptr;

    #ifdef UNIFIED_GPU_MEMORY
            // unified memory allocation
            gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&d_ptr), var_elements * var_size * allocSize))
    #else
            // non unified memory allocation
            gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_ptr), var_elements * var_size * allocSize));
    #endif

            // store the pointer in the map
            memory_map.insert(CUDAMemoryMap::value_type(sub_var_name, d_ptr));
        }
    }
}
void CUDASubAgentStateList::releaseDeviceAgentList(CUDAMemoryMap& memory_map) {
    // for each device pointer in the cuda memory map we need to free these
    for (const auto& mm : memory_map) {
        // get the variable name
        std::string sub_var_name = mm.first;
        auto map = mapping->variables.find(sub_var_name);
        // Only handle variables which are not mapped
        if (map == mapping->variables.end()) {
            // free the memory on the device
            gpuErrchk(cudaFree(mm.second));
        }
    }
}

void CUDASubAgentStateList::setLists(const std::string &var_name, void *list, void *swap_list, void *new_list) {
    if (list)
        d_list[var_name] = list;
    if (swap_list)
        d_swap_list[var_name] = swap_list;
    if (new_list)
        d_new_list[var_name] = new_list;
    // Update Condition lists
    if (list || swap_list) {
        const auto &var = agent.getAgentDescription().variables.at(var_name);
        if (list)
            condition_d_list[var_name] = reinterpret_cast<char*>(list) + (condition_state * var.type_size * var.elements);
        if (swap_list)
            condition_d_swap_list[var_name] = reinterpret_cast<char*>(swap_list) + (condition_state * var.type_size * var.elements);
    }
}
