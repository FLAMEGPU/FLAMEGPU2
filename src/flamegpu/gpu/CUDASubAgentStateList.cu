#include "flamegpu/gpu/CUDASubAgentStateList.h"

#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/gpu/CUDASubAgent.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/AgentData.h"

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
void CUDASubAgentStateList::resize(bool retain_d_list_data) {
    // Resize all variables which have not been mapped
    resizeDeviceAgentList(d_list, agent.getMaximumListSize(), retain_d_list_data);
    resizeDeviceAgentList(d_swap_list, agent.getMaximumListSize(), false);
    if (!condition_d_list.size()) {
        // Init condition state lists (late, as size was 0 at constructor)
        // Some of these might fail silently, as master_agent inits mapped vars
        // But it doesn't matter, setConditionState() updates them all
        for (const auto &c : d_list)
            condition_d_list.emplace(c);
        for (const auto &c : d_swap_list)
            condition_d_swap_list.emplace(c);
    }
    setConditionState(condition_state);  // Update pointers in condition state list
}
void CUDASubAgentStateList::setAgentData(const AgentStateMemory &state_memory) {
    CUDAAgentStateList::setAgentData(state_memory);
    // Use normal resize, but also change the agent count of the mapped state vector
    master_list->setCUDAStateListSize(current_list_size);
    // This is written with the assumption that condition_state is 0 (setAgentData() makes it 0)
    assert(master_list->getCUDAStateListSize() == current_list_size);
}
unsigned int CUDASubAgentStateList::scatter(const unsigned int &streamId, const unsigned int out_offset, const ScatterMode &mode) {
    return master_list->scatter(streamId, out_offset, mode);
}
void CUDASubAgentStateList::allocateDeviceAgentList(CUDAMemoryMap &memory_map, const unsigned int &allocSize) {
    // for each variable allocate a device array and add to map
    for (const auto &mm : agent.getAgentDescription().variables) {
        // get the variable name
        const std::string sub_var_name = mm.first;
        const auto map = mapping->variables.find(sub_var_name);
        // Only handle variables which are not mapped, unless we're handling new list
        if (map == mapping->variables.end() || memory_map == d_new_list) {
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
        // Only handle variables which are not mapped, unless we're handling new list
        if (map == mapping->variables.end() || memory_map == d_new_list) {
            // free the memory on the device
            gpuErrchk(cudaFree(mm.second));
        }
    }
}

void CUDASubAgentStateList::setLists(const std::string &var_name, void *list, void *swap_list) {
    if (list)
        d_list[var_name] = list;
    if (swap_list)
        d_swap_list[var_name] = swap_list;
    // Update Condition lists
    if (list || swap_list) {
        const auto &var = agent.getAgentDescription().variables.at(var_name);
        if (list)
            condition_d_list[var_name] = reinterpret_cast<char*>(list) + (condition_state * var.type_size * var.elements);
        if (swap_list)
            condition_d_swap_list[var_name] = reinterpret_cast<char*>(swap_list) + (condition_state * var.type_size * var.elements);
    }
    if (dependent_state) {
        dependent_state->setLists(var_name, list, swap_list);
    }
}

void CUDASubAgentStateList::resizeDeviceAgentList(CUDAMemoryMap &agent_list, const unsigned int &newSize, bool copyData) {
    // for each variable allocate a device array and add to map
    for (const auto &mm : agent.getAgentDescription().variables) {
        // get the variable name
        const std::string sub_var_name = mm.first;
        const auto map = mapping->variables.find(sub_var_name);
        // Only handle variables which are not mapped, unless we're handling new list
        if (map == mapping->variables.end() || agent_list == d_new_list) {
            const auto &var = agent.getAgentDescription().variables.at(sub_var_name);
            const size_t &type_size = var.type_size * var.elements;
            const size_t alloc_size = type_size * newSize;
            {
                // Allocate bigger new memory
                void * new_ptr = nullptr;
    #ifdef UNIFIED_GPU_MEMORY
                // unified memory allocation
                gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&new_ptr), alloc_size))
    #else
                // non unified memory allocation
                gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&new_ptr), alloc_size));
    #endif
                auto it = agent_list.find(sub_var_name);
                if (it != agent_list.end()) {
                    void *old_ptr = it->second;
                    if (copyData) {
                        const size_t active_len = current_list_size * type_size;
                        const size_t inactive_len = (agent.getMaximumListSize() - current_list_size) * type_size;
                        // Copy across old data
                        gpuErrchk(cudaMemcpy(new_ptr, old_ptr, active_len, cudaMemcpyDeviceToDevice));
                        // Zero remaining new data
                        gpuErrchk(cudaMemset(reinterpret_cast<char*>(new_ptr) + active_len, 0, inactive_len));
                    } else {
                        // Zero remaining new data
                        gpuErrchk(cudaMemset(new_ptr, 0, alloc_size));
                    }
                    // Release old data
                    gpuErrchk(cudaFree(old_ptr));
                    // Replace old data in class member vars
                    it->second = new_ptr;
                } else {
                    // If agent list is yet to be allocated just add it straight in
                    agent_list.emplace(sub_var_name, new_ptr);
                    // Zero remaining new data
                    gpuErrchk(cudaMemset(new_ptr, 0, alloc_size));
                }
            }
        }
    }
}

void CUDASubAgentStateList::appendScatterMaps(CUDAMemoryMap &merge_d_list, CUDAMemoryMap &merge_d_swap_list, VariableMap &merge_var_list) {
    const VariableMap &vars = agent.getAgentDescription().variables;
    // for each variable allocate a device array and add to map
    for (const auto &mm : vars) {
        // get the variable name
        const std::string sub_var_name = mm.first;
        const auto map = mapping->variables.find(sub_var_name);
        // Only handle variables which are not mapped
        if (map == mapping->variables.end()) {
            const std::string sub_var_merge_name = "_" + mm.first;  // Prepend reserved word _, to avoid clashes
            // Documentation isn't too clear, insert *should* throw an exception if we have a clash
            merge_d_list.insert({sub_var_merge_name, d_list.at(sub_var_name)});
            merge_d_swap_list.insert({sub_var_merge_name, d_swap_list.at(sub_var_name)});
            merge_var_list.insert({sub_var_merge_name, vars.at(sub_var_name)});
        }
    }
    if (dependent_state) {
        dependent_state->appendScatterMaps(merge_d_list, merge_d_swap_list, merge_var_list);
    }
}

void CUDASubAgentStateList::swap() {
    std::swap(d_list, d_swap_list);
    std::swap(condition_d_list, condition_d_swap_list);
    current_list_size = master_list->getCUDATrueStateListSize();
    if (dependent_state) {
        dependent_state->swap();
    }
}
void CUDASubAgentStateList::addAgents(const unsigned int &new_births, const unsigned int &streamId) {
    assert(agent.getMaximumListSize() >= current_list_size + new_births);
    CUDAScatter &scatter = CUDAScatter::getInstance(streamId);
    scatter.broadcastInit(
        agent.getAgentDescription().variables,
        d_list,
        new_births, getCUDAStateListSize());
    // Init new of dependent
    if (dependent_state) {
        dependent_state->addAgents(new_births, streamId);
    }
    current_list_size += new_births;
}
