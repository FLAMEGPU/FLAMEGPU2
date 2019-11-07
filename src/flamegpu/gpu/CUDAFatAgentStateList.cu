#include "flamegpu/gpu/CUDAFatAgentStateList.h"
#include "flamegpu/gpu/CUDAScatter.h"

CUDAFatAgentStateList::CUDAFatAgentStateList(const AgentData& description)
    : aliveAgents(0)
    , disabledAgents(0)
    , bufferLen(0) {
    // Initial statelist, must be from agent index 0
    // State lists begin unallocated, allocated on first use
    for (const auto &v : description.variables) {
        AgentVariable variable = {0u, v.first};
        variables.emplace(variable, std::make_shared<VariableBuffer>(v.second.type, v.second.type_size, v.second.default_value, v.second.elements));
    }
    // All initial variables are unique
    for (const auto &s : variables)
        variables_unique.push_back(s.second);
}
CUDAFatAgentStateList::CUDAFatAgentStateList(const CUDAFatAgentStateList& other)
    : aliveAgents(other.aliveAgents)
    , disabledAgents(other.disabledAgents)
    , bufferLen(0) {
    assert(other.bufferLen == 0);
    std::unordered_map<void*, std::shared_ptr<VariableBuffer>> var_map;
    // Copy all unique variables, create a temporary map of old unique var to new unique var
    for (const auto &v : other.variables_unique) {
        assert(v->data == nullptr);
        // Ensure that copy constructor is used
        auto t_var = std::make_shared<VariableBuffer>(*v.get());
        variables_unique.push_back(t_var);
        var_map.emplace(v.get(), t_var);
    }
    // Using var map, solve variable pairings
    for (const auto &v : other.variables) {
        variables.emplace(v.first, var_map.at(v.second.get()));
    }
}
CUDAFatAgentStateList::~CUDAFatAgentStateList() {
    for (const auto &buff : variables_unique) {
        gpuErrchk(cudaFree(buff->data));
        gpuErrchk(cudaFree(buff->data_swap));
    }
}
void CUDAFatAgentStateList::addSubAgentVariables(
  const AgentData &description,
  const unsigned int &master_fat_index,
  const unsigned int &sub_fat_index,
  const std::shared_ptr<SubAgentData> &mapping) {
    for (const auto &v : description.variables) {
        const auto &mapped = mapping->variables.find(v.first);
        AgentVariable sub_var = {sub_fat_index, v.first};
        if (mapped != mapping->variables.end()) {
            // Variable is mapped, so use existing variable
            AgentVariable master_var = {master_fat_index, mapped->second};
            variables.emplace(sub_var, variables.at(master_var));
        } else {
            // Variable is not mapped, so create new variable
            auto t_buff = std::make_shared<VariableBuffer>(v.second.type, v.second.type_size, v.second.default_value, v.second.elements);
            variables.emplace(sub_var, t_buff);
            variables_unique.push_back(t_buff);
        }
    }
}
std::shared_ptr<VariableBuffer> CUDAFatAgentStateList::getVariableBuffer(const unsigned int &fat_index, const std::string &name) {
    const AgentVariable variable = {fat_index, name};
    return variables.at(variable);
}
void CUDAFatAgentStateList::resize(const unsigned int &minSize, const bool &retainData) {
    // If already big enough return
    if (minSize <= bufferLen)
        return;

    // else, decide new size
    unsigned int newSize = bufferLen > 1024 ? bufferLen : 1024;
    while (newSize < minSize)
        newSize = static_cast<unsigned int>(newSize * 1.25f);

    // Resize all buffers in fat state list
    for (auto &buff : variables_unique) {
        const size_t var_size = buff->type_size * buff->elements;
        const size_t buff_size = var_size * newSize;
        // Free old swap buffer
        gpuErrchk(cudaFree(buff->data_swap));
        // Allocate new buffer to swap
        gpuErrchk(cudaMalloc(&buff->data_swap, buff_size));
        // Copy old data to new buffer in swap
        if (retainData && buff->data) {
            const size_t active_len = aliveAgents * var_size;
            // const size_t inactive_len = (newSize - aliveAgents) * var_size;
            // Copy across old data (TODO: We could improve this by doing a scatter for all variables at once)
            gpuErrchk(cudaMemcpy(buff->data_swap, buff->data, active_len, cudaMemcpyDeviceToDevice));
            // Zero remaining new data (This will be overwritten before use, so redundant)
            // gpuErrchk(cudaMemset(reinterpret_cast<char*>(buff->data_swap) + active_len, 0, inactive_len));
        } else {
            // Zero remaining new data (This will be overwritten before use, so redundant)
            // gpuErrchk(cudaMemset(buff->data_swap, 0, buff_size));
        }
        // Swap buffers
        std::swap(buff->data_swap, buff->data);
        // Free old swap buffer
        gpuErrchk(cudaFree(buff->data_swap));
        // Allocate new buffer to swap
        gpuErrchk(cudaMalloc(&buff->data_swap, buff_size));
        // Update condition list
        assert(disabledAgents == 0);
        buff->data_condition = buff->data;
    }

    // Update buffer len
    bufferLen = newSize;

    // Clear count
    if (!retainData) {
        aliveAgents = 0;
        disabledAgents = 0;
    }
}
unsigned int CUDAFatAgentStateList::getSize() const {
    return aliveAgents - disabledAgents;
}
unsigned int CUDAFatAgentStateList::getSizeWithDisabled() const {
    return aliveAgents;
}
unsigned int CUDAFatAgentStateList::getAllocatedSize() const {
    return bufferLen;
}
void CUDAFatAgentStateList::setAgentCount(const unsigned int &newCount, const bool &resetDisabled) {
    if ((resetDisabled && newCount > bufferLen) || (!resetDisabled && (newCount + disabledAgents> bufferLen))) {
        THROW InvalidMemoryCapacity("Agent count will exceed allocated buffer size, "
        "in CUDAFatAgentStateList::setAgentCount()\n");
    }
    if (resetDisabled) {
        disabledAgents = 0;
    }
    aliveAgents = disabledAgents + newCount;
}
unsigned int CUDAFatAgentStateList::scatterDeath(const unsigned int &streamId) {
    // Build scatter data
    std::vector<CUDAScatter::ScatterData> sd;
    for (const auto &v : variables_unique) {
        char *in_p = reinterpret_cast<char*>(v->data);
        char *out_p = reinterpret_cast<char*>(v->data_swap);
        sd.push_back({ v->type_size * v->elements, in_p, out_p });
        // Pre swap stored pointers
        std::swap(v->data, v->data_swap);
        // Pre update data_condition
        v->data_condition = out_p + (disabledAgents * v->type_size * v->elements);
    }
    // Perform scatter
    CUDAScatter &scatter = CUDAScatter::getInstance(streamId);
    const unsigned int living_agents = scatter.scatter(
        CUDAScatter::Type::AgentDeath, sd,
        aliveAgents, 0, false, disabledAgents);
    // Update size
    assert(living_agents <= bufferLen);
    aliveAgents = living_agents;

    return living_agents;
}
unsigned int CUDAFatAgentStateList::scatterAgentFunctionConditionFalse(const unsigned int &streamId) {
    // This makes no sense if we have disabled agents (it's suppose to reorder to create disabled agents)
    assert(disabledAgents == 0);
    // Build scatter data
    std::vector<CUDAScatter::ScatterData> sd;
    for (const auto &v : variables_unique) {
        char *in_p = reinterpret_cast<char*>(v->data);
        char *out_p = reinterpret_cast<char*>(v->data_swap);
        sd.push_back({ v->type_size * v->elements, in_p, out_p });
    }
    // Perform scatter
    CUDAScatter &scatter = CUDAScatter::getInstance(streamId);
    const unsigned int scattered_agents = scatter.scatter(
        CUDAScatter::Type::AgentDeath, sd,
        aliveAgents, 0, false, disabledAgents);
    return scattered_agents;
}
unsigned int CUDAFatAgentStateList::scatterAgentFunctionConditionTrue(const unsigned int &conditionFailCount, const unsigned int &streamId) {
    // This makes no sense if we have disabled agents (it's suppose to reorder to create disabled agents)
    assert(disabledAgents == 0);
    // Build scatter data
    std::vector<CUDAScatter::ScatterData> sd;
    for (const auto &v : variables_unique) {
        char *in_p = reinterpret_cast<char*>(v->data);
        char *out_p = reinterpret_cast<char*>(v->data_swap);
        sd.push_back({ v->type_size * v->elements, in_p, out_p });
        // Pre swap stored pointers
        std::swap(v->data, v->data_swap);
        // Pre update data_condition
        v->data_condition = out_p + (conditionFailCount * v->type_size * v->elements);
    }
    // Perform scatter
    CUDAScatter &scatter = CUDAScatter::getInstance(streamId);
    const unsigned int scattered_agents = scatter.scatter(
        CUDAScatter::Type::AgentDeath, sd,
        aliveAgents, conditionFailCount, true, disabledAgents);
    // Update disabled agents count
    disabledAgents = conditionFailCount;
    return scattered_agents;
}
void CUDAFatAgentStateList::setDisabledAgents(const unsigned int numberOfDisabled) {
    assert(numberOfDisabled <= aliveAgents);
    disabledAgents = numberOfDisabled;
    // update data_condition for each unique variable
    for (const auto &v : variables_unique) {
        char *data_p = reinterpret_cast<char*>(v->data);
        v->data_condition = data_p + (numberOfDisabled * v->type_size * v->elements);
    }
}
void CUDAFatAgentStateList::initVariables(std::set<std::shared_ptr<VariableBuffer>> &exclusionSet, const unsigned int initCount, const unsigned initOffset, const unsigned int &streamId) {
    if (initCount && exclusionSet.size()) {
        assert(initCount + initOffset <= bufferLen);
        std::list<std::shared_ptr<VariableBuffer>> initVars;
        // Build list of init vars (to save repeating this process), and calculate memory requirements
        for (const auto &v : variables_unique) {
            if (exclusionSet.find(v) == exclusionSet.end()) {
                initVars.push_back(v);
            }
        }
        // Perform scatter
        CUDAScatter &scatter = CUDAScatter::getInstance(streamId);
        scatter.broadcastInit(initVars, initCount, initOffset);
    }
}

std::list<std::shared_ptr<VariableBuffer>> &CUDAFatAgentStateList::getUniqueVariables() { return variables_unique; }

void CUDAFatAgentStateList::swap(CUDAFatAgentStateList*other) {
    std::swap(aliveAgents, other->aliveAgents);
    std::swap(disabledAgents, other->disabledAgents);
    std::swap(bufferLen, other->bufferLen);
    for (auto a = variables_unique.begin(), b=other->variables_unique.begin(); a != variables_unique.end() && b != other->variables_unique.end(); ++a, ++b) {
        (*a)->swap(b->get());
    }
}
