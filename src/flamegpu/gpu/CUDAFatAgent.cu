#include "flamegpu/gpu/CUDAFatAgent.h"

#include "flamegpu/gpu/CUDAScatter.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

CUDAFatAgent::CUDAFatAgent(const AgentData& description)
  : mappedAgentCount(0) {
    for (const std::string &s : description.states) {
        // allocate memory for each state list by creating a new Agent State List
        AgentState state = {mappedAgentCount, s};
        states.emplace(state, std::make_shared<CUDAFatAgentStateList>(description));
    }
    mappedAgentCount++;
    // All initial states are unique
    for (const auto &s : states)
        states_unique.insert(s.second);
}
CUDAFatAgent::~CUDAFatAgent() {
    for (auto &b : d_newLists) {
        gpuErrchk(cudaFree(b.data));
    }
    d_newLists.clear();
}
std::unordered_map<std::string, std::shared_ptr<CUDAFatAgentStateList>> CUDAFatAgent::getStateMap(const unsigned int &fat_index) {
    std::unordered_map<std::string, std::shared_ptr<CUDAFatAgentStateList>> rtn;
    // For each state
    for (const auto &s : states) {
        // If it corresponds to the correct agent
        if (s.first.agent == fat_index) {
            // Store in map
            rtn.emplace(s.first.state, s.second);
        }
    }
    return rtn;
}
void CUDAFatAgent::addSubAgent(
  const AgentData &description,
  const unsigned int &master_fat_index,
  const std::shared_ptr<SubAgentData> &mapping) {
    assert(states.size());
    assert(states_unique.size());
    // Handle agent states
    for (const std::string &s : description.states) {
        const auto &mapped = mapping->states.find(s);
        AgentState sub_state = {mappedAgentCount, s};
        if (mapped != mapping->states.end()) {
            // State is mapped, so use existing state
            AgentState master_state = {master_fat_index, mapped->second};
            states.emplace(sub_state, states.at(master_state));
        } else {
            // State is not mapped, so create new state, simply clone any existing state
            // This works as the existing state should be uninitialised buffer per variable
            auto cloned_state = std::make_shared<CUDAFatAgentStateList>(*states_unique.begin()->get());
            states.emplace(sub_state, cloned_state);
            states_unique.insert(cloned_state);
        }
        // allocate memory for each state list by creating a new Agent State List
        AgentState state = {mappedAgentCount, s};
        states.emplace(state, std::make_shared<CUDAFatAgentStateList>(description));
    }
    // Handle agent variables
    for (auto &state : states_unique) {
        // For each unique state of the fat agent
        // Add sub_agent vars
        // This includes some which are 'redundant', as they may only be used if an agent transfers between states
        // to prevent loss of data
        state->addSubAgentVariables(description, master_fat_index, mappedAgentCount, mapping);
    }
    mappedAgentCount++;
}

void CUDAFatAgent::processDeath(const unsigned int &agent_fat_id, const std::string &state_name, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    auto sm = states.find({agent_fat_id, state_name});
    if (sm == states.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAFatAgent::processDeath()",
            "?", state_name.c_str());
    }

    CUDAScanCompactionConfig &scanCfg = scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId);
    const unsigned int agent_count = sm->second->getSize();
    // Resize cub (if required)
    if (agent_count > scanCfg.cub_temp_size_max_list_size) {
        if (scanCfg.hd_cub_temp) {
            gpuErrchk(cudaFree(scanCfg.hd_cub_temp));
        }
        scanCfg.cub_temp_size = 0;
        gpuErrchk(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scanCfg.cub_temp_size,
            scanCfg.d_ptrs.scan_flag,
            scanCfg.d_ptrs.position,
            sm->second->getAllocatedSize() + 1,
            stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        gpuErrchk(cudaMalloc(&scanCfg.hd_cub_temp,
            scanCfg.cub_temp_size));
        scanCfg.cub_temp_size_max_list_size = sm->second->getAllocatedSize();
    }
    gpuErrchk(cub::DeviceScan::ExclusiveSum(
        scanCfg.hd_cub_temp,
        scanCfg.cub_temp_size,
        scanCfg.d_ptrs.scan_flag,
        scanCfg.d_ptrs.position,
        agent_count + 1,
        stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    // Scatter
    sm->second->scatterDeath(scatter, streamId, stream);
}

void CUDAFatAgent::transitionState(const unsigned int &agent_fat_id, const std::string &_src, const std::string &_dest, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    // Optionally process state transition
    if (_src != _dest) {
        auto src = states.find({agent_fat_id, _src});
        if (src == states.end()) {
            THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
                "in CUDAFatAgent::transitionState()",
                "?", _src.c_str());
        }
        // If src list is empty we can skip
        if (src->second->getSizeWithDisabled() == 0)
            return;
        auto dest = states.find({agent_fat_id, _dest});
        if (dest == states.end()) {
            THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
                "in CUDAFatAgent::transitionState()",
                "?", _dest.c_str());
        }
        // If dest list is empty and we are not in an agent function condition, we can swap the lists
        if (dest->second->getSizeWithDisabled() == 0 && src->second->getSize() == src->second->getSizeWithDisabled()) {
            // This swaps the master_lists entire states (std::swap would only swap pointers in fat_agent, we need to swap components to update copies of shared_ptr)
            states.at({agent_fat_id, _src})->swap(states.at({agent_fat_id, _dest}).get());
        } else {
            // Otherwise we must perform a scatter all operation
            // Resize destination list
            dest->second->resize(src->second->getSize() + dest->second->getSizeWithDisabled(), true);
            // Build scatter data
            // It's assumed that each CUDAFatAgentStatelist has it's unique variables list in the same order, so we can map across from 1 to other
            auto &src_v = src->second->getUniqueVariables();
            auto &dest_v = dest->second->getUniqueVariables();
            std::vector<CUDAScatter::ScatterData> sd;
            for (auto src_it = src_v.begin(), dest_it = dest_v.begin(); src_it != src_v.end() && dest_it != dest_v.end(); ++src_it, ++dest_it) {
                char *in_p = reinterpret_cast<char*>((*src_it)->data_condition);
                char *out_p = reinterpret_cast<char*>((*dest_it)->data);
                sd.push_back({ (*src_it)->type_size * (*src_it)->elements, in_p, out_p });
                assert((*src_it)->type_size == (*dest_it)->type_size);
                assert((*src_it)->elements == (*dest_it)->elements);
            }
            // Perform scatter
            scatter.scatterAll(streamId, stream, sd, src->second->getSize(), dest->second->getSizeWithDisabled());
            // Update list sizes
            dest->second->setAgentCount(dest->second->getSize() + src->second->getSize());
            src->second->setAgentCount(0);
        }
    }
}

void CUDAFatAgent::processFunctionCondition(const unsigned int &agent_fat_id, const std::string &state_name, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    auto sm = states.find({agent_fat_id, state_name});
    if (sm == states.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAFatAgent::processFunctionCondition()",
            "?", state_name.c_str());
    }

    CUDAScanCompactionConfig &scanCfg = scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_DEATH, streamId);
    unsigned int agent_count = sm->second->getSize();
    // Resize cub (if required)
    if (agent_count > scanCfg.cub_temp_size_max_list_size) {
        if (scanCfg.hd_cub_temp) {
            gpuErrchk(cudaFree(scanCfg.hd_cub_temp));
        }
        scanCfg.cub_temp_size = 0;
        gpuErrchk(cub::DeviceScan::ExclusiveSum(
            nullptr,
            scanCfg.cub_temp_size,
            scanCfg.d_ptrs.scan_flag,
            scanCfg.d_ptrs.position,
            sm->second->getAllocatedSize() + 1));
        gpuErrchk(cudaMalloc(&scanCfg.hd_cub_temp,
            scanCfg.cub_temp_size));
        scanCfg.cub_temp_size_max_list_size = sm->second->getAllocatedSize();
    }
    // Perform scan (agent function conditions use death flag scan compact arrays as there is no overlap in use)
    gpuErrchk(cub::DeviceScan::ExclusiveSum(
        scanCfg.hd_cub_temp,
        scanCfg.cub_temp_size,
        scanCfg.d_ptrs.scan_flag,
        scanCfg.d_ptrs.position,
        agent_count + 1,
        stream));
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));
    // Use scan results to sort false agents into start of list (and don't swap buffers)
    const unsigned int conditionFailCount = sm->second->scatterAgentFunctionConditionFalse(scatter, streamId, stream);
    // Invert scan
    CUDAScatter::InversionIterator ii = CUDAScatter::InversionIterator(scanCfg.d_ptrs.scan_flag);
    cudaMemsetAsync(scanCfg.d_ptrs.position, 0, sizeof(unsigned int)*(agent_count + 1), stream);
    gpuErrchk(cub::DeviceScan::ExclusiveSum(
        scanCfg.hd_cub_temp,
        scanCfg.cub_temp_size,
        ii,
        scanCfg.d_ptrs.position,
        agent_count + 1,
        stream));
    gpuErrchkLaunch();
    gpuErrchk(cudaStreamSynchronize(stream));
    // Use inverted scan results to sort true agents into end of list (and swap buffers)
    const unsigned int conditionpassCount = sm->second->scatterAgentFunctionConditionTrue(conditionFailCount, scatter, streamId, stream);
    assert(agent_count == conditionpassCount + conditionFailCount);
}

void CUDAFatAgent::setConditionState(const unsigned int &agent_fat_id, const std::string &state_name, const unsigned int numberOfDisabled) {
    // check the cuda agent state map to find the correct state list for functions starting state
    auto sm = states.find({agent_fat_id, state_name});
    if (sm == states.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAFatAgent::setConditionState()",
            "?", state_name.c_str());
    }
    sm->second->setDisabledAgents(numberOfDisabled);
}

void *CUDAFatAgent::allocNewBuffer(const size_t &total_agent_size, const unsigned int &new_agents, const size_t &varCount) {
    std::lock_guard<std::mutex> guard(d_newLists_mutex);
    const size_t SIZE_REQUIRED = total_agent_size * new_agents + (varCount-1) * 8;
    const size_t ALLOCATION_SIZE = static_cast<size_t>(SIZE_REQUIRED * 1.25);  // Premept larger buffer req, reduce reallocations
    // Find the smallest existing buffer with enough size
    for (auto &b : d_newLists) {
        if (b.size >= SIZE_REQUIRED && !b.in_use) {
            // Buffer is suitable
            NewBuffer my_b = b;
            // Erase and reinsert to d_newLists to mark as in use
            d_newLists.erase(b);
            my_b.in_use = true;
            d_newLists.insert(my_b);
            // Return buffer
            return my_b.data;
        }
    }
    // Find the smallest free list and resize it to be big enough
    for (auto &b : d_newLists) {
        if (!b.in_use) {
            // Buffer is suitable
            NewBuffer my_b = b;
            // Erase and resize/reinsert to d_newLists to mark as in use
            d_newLists.erase(b);
            gpuErrchk(cudaFree(my_b.data));
            gpuErrchk(cudaMalloc(&my_b.data, ALLOCATION_SIZE));
            my_b.size = ALLOCATION_SIZE;
            my_b.in_use = true;
            d_newLists.insert(my_b);
            // Return buffer
            return my_b.data;
        }
    }
    // No existing buffer available, so create a new one
    NewBuffer my_b;
    gpuErrchk(cudaMalloc(&my_b.data, ALLOCATION_SIZE));
    my_b.size = ALLOCATION_SIZE;
    my_b.in_use = true;
    d_newLists.insert(my_b);
    return my_b.data;
}

void CUDAFatAgent::freeNewBuffer(void *buff) {
    std::lock_guard<std::mutex> guard(d_newLists_mutex);
    // Find the right buffer
    for (auto &b : d_newLists) {
        if (b.data == buff) {
            assert(b.in_use);
            // Erase and reinsert to d_newLists to mark as free
            NewBuffer my_b = b;
            d_newLists.erase(b);
            my_b.in_use = false;
            d_newLists.insert(my_b);
            return;
        }
    }
    assert(false);
}
unsigned int CUDAFatAgent::getMappedAgentCount() const { return mappedAgentCount; }
