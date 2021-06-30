#include "flamegpu/gpu/CUDAAgentStateList.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/runtime/HostNewAgentAPI.h"
#include "flamegpu/exception/FLAMEGPUException.h"

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#endif  // _MSC_VER
#include <cub/cub.cuh>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

namespace flamegpu {

CUDAAgentStateList::CUDAAgentStateList(
    const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
    CUDAAgent& cuda_agent,
    const unsigned int &_fat_index,
    const AgentData& description,
    bool _isSubStateList)
    : fat_index(_fat_index)
    , agent(cuda_agent)
    , parent_list(fat_list)
    , isSubStateList(_isSubStateList) {
    // For each agent variable, take a copy of the shared pointer, store it
    for (auto var : description.variables) {
        variables.emplace(var.first, fat_list->getVariableBuffer(fat_index, var.first));
    }
}
CUDAAgentStateList::CUDAAgentStateList(
    const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
    CUDAAgent& cuda_agent,
    const unsigned int &_fat_index,
    const AgentData& description,
    bool _isSubStateList,
    const SubAgentData::Mapping &varMap)
    : CUDAAgentStateList(fat_list, cuda_agent, _fat_index, description, _isSubStateList) {
    // Build a list of variables not present in the mapping
    // These are not mapped to parent agent, therefore they must be reset when CUDASimulation::simulate() is called
    for (auto var : variables) {
        if (varMap.find(var.first)== varMap.end()) {
            unmappedBuffers.push_back(var.second);
        }
    }
}
void CUDAAgentStateList::resize(const unsigned int &minimumSize, const bool &retainData) {
    parent_list->resize(minimumSize, retainData);
}
unsigned int CUDAAgentStateList::getSize() const {
    return parent_list->getSize();
}
/**
 * Returns the maximum number of agents that can be stored based on the current buffer allocations
 */
unsigned int CUDAAgentStateList::getAllocatedSize() const {
    return parent_list->getAllocatedSize();
}
void *CUDAAgentStateList::getVariablePointer(const std::string &variable_name) {
    // check the cuda agent state map to find the correct state list for functions starting state
    auto var = variables.find(variable_name);

    if (var == variables.end()) {
        THROW exception::InvalidAgentVar("Error: Agent ('%s') variable ('%s') was not found "
            "in CUDAAgentStateList::getVariablePointer()",
            agent.getAgentDescription().name.c_str(), variable_name.c_str());
    }

    return var->second->data_condition;
}
void CUDAAgentStateList::setAgentData(const AgentVector& population, CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream) {
    // Validate AgentData matches
    if (!population.matchesAgentType(agent.getAgentDescription())) {
        THROW exception::InvalidCudaAgentDesc("Agent description for agent '%s' does not match that of AgentVector, "
            "in CUDAAgentStateList::setAgentData()",
            population.getAgentName().c_str());
    }
    // Check our internal state matches or exceeds the size of the state in the agent pop
    // This will return if list already correct size
    const unsigned int data_count = population.size();
    if (data_count) {
        parent_list->resize(data_count, false);  // FALSE=Do not retain existing data
        // Initialise any buffers in the fat_agent which aren't part of the agent description
        std::set<std::shared_ptr<VariableBuffer>> exclusionSet;
        for (auto& a : variables)
            exclusionSet.insert(a.second);
        parent_list->initVariables(exclusionSet, data_count, 0, scatter, streamId, stream);
        // Copy across the required data host->device
        for (auto& _var : variables) {
            // get the variable size from agent description
            const auto& var = agent.getAgentDescription().variables.at(_var.first);
            const size_t var_size = var.type_size;
            const unsigned int  var_elements = var.elements;

            // get pointer to vector data
            const void* v_data = population.data(_var.first);

            // copy the host data to the GPU
            gpuErrchk(cudaMemcpyAsync(_var.second->data, v_data, var_elements * var_size * data_count, cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaStreamSynchronize(stream));
        }
    }
    // Update alive count etc
    parent_list->setAgentCount(data_count);
}
void CUDAAgentStateList::getAgentData(AgentVector& population) const {
    // Validate AgentData matches
    if (!population.matchesAgentType(agent.getAgentDescription())) {
        THROW exception::InvalidCudaAgentDesc("Agent description for agent '%s' does not match that of AgentVector, "
            "in CUDAAgentStateList::setAgentData()",
            population.getAgentName().c_str());
    }
    const unsigned int data_count = getSize();
    if (data_count) {
        population.internal_resize(data_count, false);
        // Copy across the required data device->host
        for (auto& _var : variables) {
            // get the variable size from agent description
            const auto& var = agent.getAgentDescription().variables.at(_var.first);
            const size_t var_size = var.type_size;
            const unsigned int  var_elements = var.elements;

            // get pointer to vector data
            // Use the const method, but const cast away the const to avoid the reserved var check
            void* v_data = const_cast<void*>(static_cast<const AgentVector&>(population).data(_var.first));

            // copy the host data to the GPU
            gpuErrchk(cudaMemcpy(v_data, _var.second->data, var_elements * var_size * data_count, cudaMemcpyDeviceToHost));
        }
    }
    population._size = data_count;  // Private AgentVector::resize() does not update size
}
void CUDAAgentStateList::scatterHostCreation(const unsigned int& newSize, char* const d_inBuff, const VarOffsetStruct & offsets, CUDAScatter & scatter, const unsigned int& streamId, const cudaStream_t & stream) {
    // Resize agent list if required
    parent_list->resize(parent_list->getSizeWithDisabled() + newSize, true);
    // Build scatter data
    std::vector<CUDAScatter::ScatterData> sd;
    for (const auto &v : variables) {
        // In this case, in is the location of first variable, but we step by inOffsetData.totalSize
        char *in_p = reinterpret_cast<char*>(d_inBuff) + offsets.vars.at(v.first).offset;
        char *out_p = reinterpret_cast<char*>(v.second->data);
        sd.push_back({ v.second->type_size * v.second->elements, in_p, out_p });
    }
    // Scatter to device
    scatter.scatterNewAgents(streamId,
        stream,
        sd,
        offsets.totalSize,
        newSize,
        parent_list->getSize());
    // Initialise any buffers in the fat_agent which aren't part of the current agent description
    // TODO: This does redundant inits, it only needs to initialise parent/master agent variables which are not mapped
    //       Sub variables will already be init everytime the submodel is executed.
    std::set<std::shared_ptr<VariableBuffer>> exclusionSet;
    for (auto &a : variables)
        exclusionSet.insert(a.second);
    parent_list->initVariables(exclusionSet, newSize, parent_list->getSize(), scatter, streamId, stream);
    // Update number of alive agents
    parent_list->setAgentCount(parent_list->getSize() + newSize);
}
void CUDAAgentStateList::scatterSort(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    parent_list->scatterSort(scatter, streamId, stream);
}
unsigned int CUDAAgentStateList::scatterNew(void * d_newBuff, const unsigned int &newSize, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    if (newSize) {
        CUDAScanCompactionConfig &scanCfg = scatter.Scan().Config(CUDAScanCompaction::Type::AGENT_OUTPUT, streamId);
        // Perform scan
        if (newSize > scanCfg.cub_temp_size_max_list_size) {
            if (scanCfg.hd_cub_temp) {
                gpuErrchk(cudaFree(scanCfg.hd_cub_temp));
            }
            scanCfg.cub_temp_size = 0;
            gpuErrchk(cub::DeviceScan::ExclusiveSum(
                nullptr,
                scanCfg.cub_temp_size,
                scanCfg.d_ptrs.scan_flag,
                scanCfg.d_ptrs.position,
                newSize + 1,
                stream));
            gpuErrchk(cudaStreamSynchronize(stream));
            gpuErrchk(cudaMalloc(&scanCfg.hd_cub_temp, scanCfg.cub_temp_size));
            scanCfg.cub_temp_size_max_list_size = newSize;
        }
        gpuErrchk(cub::DeviceScan::ExclusiveSum(
            scanCfg.hd_cub_temp,
            scanCfg.cub_temp_size,
            scanCfg.d_ptrs.scan_flag,
            scanCfg.d_ptrs.position,
            newSize + 1,
            stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        // Resize if necessary
        // @todo? this could be improved by checking scan result for the actual size, rather than max size)
        resize(parent_list->getSizeWithDisabled() + newSize, true);
        // Build scatter data
        char * d_var = static_cast<char*>(d_newBuff);

        std::vector<CUDAScatter::ScatterData> scatterdata;
        for (const auto &v : variables) {
            char *in_p = reinterpret_cast<char*>(d_var);
            char *out_p = reinterpret_cast<char*>(v.second->data_condition);
            scatterdata.push_back({ v.second->type_size * v.second->elements, in_p, out_p });
            // Prep pointer for next var
            d_var += v.second->type_size * v.second->elements * newSize;
            // 64 bit align the new buffer start
            if (reinterpret_cast<size_t>(d_var)%8) {
                d_var += 8 - (reinterpret_cast<size_t>(d_var)%8);
            }
        }
        // Perform scatter
        const unsigned int new_births = scatter.scatter(
            streamId,
            stream,
            CUDAScatter::Type::AGENT_OUTPUT,
            scatterdata,
            newSize, parent_list->getSizeWithDisabled());
        if (new_births == 0) return 0;
        // Initialise any buffers in the fat_agent which aren't part of the current agent description
        // TODO: This does redundant inits, it only needs to initialise parent/master agent variables which are not mapped
        //       Sub variables will already be init everytime the submodel is executed.
        std::set<std::shared_ptr<VariableBuffer>> exclusionSet;
        for (auto &a : variables)
            exclusionSet.insert(a.second);
        parent_list->initVariables(exclusionSet, newSize, parent_list->getSize(), scatter, streamId, stream);
        // Update number of alive agents
        parent_list->setAgentCount(parent_list->getSize() + new_births);
        return new_births;
    }
    return 0;
}
bool CUDAAgentStateList::getIsSubStatelist() {
    return isSubStateList;
}
void CUDAAgentStateList::initUnmappedVars(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream) {
    assert(parent_list->getSizeWithDisabled() == parent_list->getSize());
    if (parent_list->getSize()) {
        assert(isSubStateList);
        // If unmappedBuffers is not empty, perform broadcast init
        if (unmappedBuffers.size()) {
            scatter.broadcastInit(streamId, stream, unmappedBuffers, parent_list->getSize(), 0);
        }
    }
}
void CUDAAgentStateList::initExcludedVars(const unsigned int& count, const unsigned int& offset, CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream) {
    std::set<std::shared_ptr<VariableBuffer>> exclusionSet;
    for (auto& a : variables)
        exclusionSet.insert(a.second);
    parent_list->initVariables(exclusionSet, count, offset, scatter, streamId, stream);
}
void CUDAAgentStateList::clear() {
    parent_list->setAgentCount(0, true);
}
void CUDAAgentStateList::setAgentCount(const unsigned int& newSize) {
    parent_list->setAgentCount(newSize, false);
}
std::list<std::shared_ptr<VariableBuffer>> CUDAAgentStateList::getUnboundVariableBuffers() {
    std::set<std::shared_ptr<VariableBuffer>> exclusionSet;
    for (auto& a : variables)
        exclusionSet.insert(a.second);
    return parent_list->getBuffers(exclusionSet);
}

}  // namespace flamegpu
