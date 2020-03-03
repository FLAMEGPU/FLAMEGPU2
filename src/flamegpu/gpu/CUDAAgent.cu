/**
* @file CUDAAgent.cpp
* @authors Paul
* @date
* @brief
*
* @see
* @warning
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAAgent.h"
#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/gpu/CUDAScatter.h"

#ifdef _MSC_VER
#pragma warning(push, 3)
#include <cub/cub.cuh>
#pragma warning(pop)
#else
#include <cub/cub.cuh>
#endif

/**
* CUDAAgent class
* @brief allocates the hash table/list for agent variables and copy the list to device
*/
CUDAAgent::CUDAAgent(const AgentData& description)
    : agent_description(description)
    , state_map()
    , max_list_size(0) {
    // Regen new empty state_map
    for (const std::string &s : agent_description.states) {
        // allocate memory for each state list by creating a new Agent State List
        state_map.insert(CUDAStateMap::value_type(s, std::unique_ptr<CUDAAgentStateList>(new CUDAAgentStateList(*this))));
    }
}

/**
 * A destructor.
 * @brief Destroys the CUDAAgent object
 */
CUDAAgent::~CUDAAgent(void) {
}


/**
* @brief Returns agent description
* @param none
* @return AgentDescription object
*/
const AgentData& CUDAAgent::getAgentDescription() const {
    return agent_description;
}

void CUDAAgent::resize(const unsigned int &newSize, const unsigned int &streamId) {
    // Only grow currently
    max_list_size = max_list_size < 2 ? 2 : max_list_size;
    if (newSize > max_list_size) {
        while (max_list_size < newSize) {
            max_list_size = static_cast<unsigned int>(max_list_size * 1.5);
        }
        // Resize all items in the statemap
        for (auto &state : state_map) {
            state.second->resize();  // It auto pulls size from this->max_list_size
        }
    }
    // Notify scan flag that it might need resizing
    flamegpu_internal::CUDAScanCompaction::resize(max_list_size, flamegpu_internal::CUDAScanCompaction::AGENT_DEATH, streamId);
}
/**
* @brief Sets the population data
* @param AgentPopulation object
* @return none
*/
void CUDAAgent::setPopulationData(const AgentPopulation& population) {
    // check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    if (state_map.empty()) {
        // create map of device state lists by traversing the state list
        for (const std::string &s : agent_description.states) {
            // allocate memory for each state list by creating a new Agent State List
            state_map.insert(CUDAStateMap::value_type(s, std::unique_ptr<CUDAAgentStateList>(new CUDAAgentStateList(*this))));
        }
    }
    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    const std::string agent_name = agent_description.name;
    if ((population.getAgentDescription()) != agent_description) {
        THROW InvalidPopulationData("Error: Initial Population has a different agent description ('%s') "
            "to that which was used to initialise the CUDAAgent ('%s'). "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str(),
            agent_description.name.c_str());
    }
    // check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() > max_list_size) {
        // Resize the population exactly, setPopData is a whole population movement, no need for greedy resize
        // Unlikely to add new agents during simulation

        // Update capacity
        max_list_size = population.getMaximumStateListCapacity();
        // Drop old state_map
        state_map.clear();
        // Regen new state_map
        for (const std::string &s : agent_description.states) {
            // allocate memory for each state list by creating a new Agent State List
            state_map.insert(CUDAStateMap::value_type(s, std::unique_ptr<CUDAAgentStateList>(new CUDAAgentStateList(*this))));
        }
    }
    /**set all population data to zero*/
    zeroAllStateVariableData();

    /**copy all population data to correct state map*/
    const std::set<std::string> &sm = agent_description.states;
    for (const std::string &s : sm) {
        // get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s);

        /**check that the CUDAAgentStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end()) {
            THROW InvalidMapEntry("Error: failed to find memory allocated for agent ('%s') state ('%s') "
                "In CUDAAgent::setPopulationData() ",
                "This should never happen!",
                population.getAgentName().c_str(), s.c_str());
        }
        // copy the data from the population state memory to the state_maps CUDAAgentStateList
        i->second->setAgentData(population.getReadOnlyStateMemory(i->first));
    }
}

void CUDAAgent::getPopulationData(AgentPopulation& population) {
    // check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    if (state_map.empty()) {
        THROW InvalidPopulationData("Error: Initial population data for agent '%s' not allocated. "
            "Have you called getPopulationData()? "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str());
    }
    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    const std::string agent_name = agent_description.name;
    if (population.getAgentDescription() != agent_description) {
        THROW InvalidPopulationData("Error: Initial Population has a different agent description ('%s') "
            "to that which was used to initialise the CUDAAgent ('%s'). "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str(),
            agent_description.name.c_str());
    }
    // Resize population if it is too small
    if (population.getMaximumStateListCapacity() < getMaximumListSize())
        population.setStateListCapacity(getMaximumListSize());

    /* copy all population from correct state maps */
    const std::set<std::string> &sm = agent_description.states;
    for (const std::string &s : sm) {
        // get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s);

        /**check that the CUDAAgentStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end()) {
            THROW InvalidMapEntry("Error: failed to find memory allocated for agent ('%s') state ('%s') "
                "In CUDAAgent::setPopulationData() ",
                "This should never happen!",
                population.getAgentName().c_str(), s.c_str());
        }
        // check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
        if (population.getMaximumStateListCapacity() < i->second->getCUDAStateListSize()) {
            // This should be redundant
            THROW InvalidPopulationData("Error: Maximum population size for agent '%s' exceeds allocation. "
                "In CUDAAgent::getPopulationData()",
                population.getAgentName().c_str());
        }
        // copy the data from the population state memory to the state_maps CUDAAgentStateList
        i->second->getAgentData(population.getStateMemory(i->first));
    }
}

/**
* @brief Returns the maximum list size
* @param none
* @return maximum size list that is equal to the maxmimum population size
*/
unsigned int CUDAAgent::getMaximumListSize() const {
    return max_list_size;
}

/**
* @brief Sets all state variable data to zero
* It loops through sate maps and resets the values
* @param none
* @return none
* @warning zeroAgentData
*/
void CUDAAgent::zeroAllStateVariableData() {
    // loop through state maps and reset the values
    for (CUDAStateMapPair& s : state_map) {
        s.second->zeroAgentData();
    }
}

// this is done for all the variables for now.
void CUDAAgent::mapRuntimeVariables(const AgentFunctionData& func, const std::string &state) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::mapRuntimeVariables()",
            agent_description.name.c_str(), func.initial_state.c_str());
    }

    const Curve::VariableHash agent_hash = Curve::getInstance().variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // map using curve
        const Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());

        // get the agent variable size
        size_t size = mmp.second.type_size;

       // maximum population num
        unsigned int length = this->getMaximumListSize();
        Curve::getInstance().registerVariableByHash(var_hash + agent_hash + func_hash, d_ptr, size, length);
    }
}

void CUDAAgent::unmapRuntimeVariables(const AgentFunctionData& func) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(func.initial_state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::unmapRuntimeVariables()",
            agent_description.name.c_str(), func.initial_state.c_str());
    }

    const Curve::VariableHash agent_hash = Curve::getInstance().variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        // void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // unmap using curve
        const Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());
        Curve::getInstance().unregisterVariableByHash(var_hash + agent_hash + func_hash);
    }
}

void CUDAAgent::processDeath(const AgentFunctionData& func, const unsigned int &streamId) {
    if (func.has_agent_death) {  // Optionally process agent death
        // check the cuda agent state map to find the correct state list for functions starting state
        CUDAStateMap::const_iterator sm = state_map.find(func.initial_state);

        unsigned int agent_count = sm->second->getCUDAStateListSize();
        // Resize cub (if required)
        if (agent_count > flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size_max_list_size) {
            if (flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp) {
                gpuErrchk(cudaFree(flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp));
            }
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.scan_flag,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.position,
                max_list_size + 1);
            gpuErrchk(cudaMalloc(&flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size));
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size_max_list_size = max_list_size;
        }
        cub::DeviceScan::ExclusiveSum(
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.scan_flag,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.position,
            agent_count + 1);

        // Scatter
        sm->second->scatter(streamId, CUDAAgentStateList::ScatterMode::Death);
    }
}
const std::unique_ptr<CUDAAgentStateList> &CUDAAgent::getAgentStateList(const std::string &state_name) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(state_name);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::getAgentStateList()",
            agent_description.name.c_str(), state_name.c_str());
    }
    return sm->second;
}

void* CUDAAgent::getStateVariablePtr(const std::string& state_name, const std::string& variable_name) {
    return getAgentStateList(state_name)->getAgentListVariablePointer(variable_name);
}

ModelData::size_type CUDAAgent::getStateSize(const std::string& state_name) const {
    return getAgentStateList(state_name)->getCUDAStateListSize();
}

void CUDAAgent::processFunctionCondition(const AgentFunctionData& func, const unsigned int &streamId) {
    if (func.condition) {  // Optionally process agent death
        // check the cuda agent state map to find the correct state list for functions starting state
        CUDAStateMap::const_iterator sm = state_map.find(func.initial_state);

        unsigned int agent_count = sm->second->getCUDAStateListSize();
        // Resize cub (if required)
        if (agent_count > flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size_max_list_size) {
            if (flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp) {
                gpuErrchk(cudaFree(flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp));
            }
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.scan_flag,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.position,
                max_list_size + 1);
            gpuErrchk(cudaMalloc(&flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp,
                flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size));
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size_max_list_size = max_list_size;
        }
        // Perform scan
        cub::DeviceScan::ExclusiveSum(
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.scan_flag,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.position,
            agent_count + 1);
        gpuErrchkLaunch();
        // Use scan results to sort false agents into start of list (and don't swap buffers)
        const unsigned int conditionFailCount = sm->second->scatter(streamId, 0, CUDAAgentStateList::FunctionCondition);
        // Invert scan
        CUDAScatter::InversionIterator ii = CUDAScatter::InversionIterator(flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.scan_flag);
        cudaMemset(flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.position, 0, sizeof(unsigned int)*(agent_count + 1));
        cub::DeviceScan::ExclusiveSum(
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].hd_cub_temp,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].cub_temp_size,
            ii,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_DEATH][streamId].d_ptrs.position,
            agent_count + 1);
        gpuErrchkLaunch();
        // Use inverted scan results to sort true agents into end of list (and swap buffers)
        const unsigned int conditionpassCount = sm->second->scatter(streamId, conditionFailCount, CUDAAgentStateList::FunctionCondition2);
        assert(agent_count == conditionpassCount + conditionFailCount);
        // Set agent function condition state
        sm->second->setConditionState(conditionFailCount);
    }
}
void CUDAAgent::clearFunctionConditionState(const std::string &state) {
        // check the cuda agent state map to find the correct state list for functions starting state
        CUDAStateMap::const_iterator sm = state_map.find(state);
        sm->second->setConditionState(0);
}

void CUDAAgent::transitionState(const std::string &_src, const std::string &_dest, const unsigned int &streamId) {
    if (_src != _dest) {
        CUDAStateMap::const_iterator src = state_map.find(_src);
        CUDAStateMap::const_iterator dest = state_map.find(_dest);
        if (src == state_map.end()) {
            THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
                "in CUDAAgent::transition_state()",
                agent_description.name.c_str(), _src.c_str());
        }
        if (dest == state_map.end()) {
            THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
                "in CUDAAgent::transition_state()",
                agent_description.name.c_str(), _dest.c_str());
        }
        // If src list is empty we can skip
        if (src->second->getCUDATrueStateListSize() == 0)
            return;
        // If dest list is empty and we are not in an gent function condition, we can swap the lists
        if (dest->second->getCUDATrueStateListSize() == 0 && src->second->getCUDAStateListSize() == src->second->getCUDATrueStateListSize()) {
            swap(state_map.at(_src), state_map.at(_dest));
            assert(state_map.find(_src)->second->getCUDAStateListSize() == 0);
        } else {  // Otherwise we must perform a scatter all operation
            auto &cs = CUDAScatter::getInstance(streamId);
            cs.scatterAll(agent_description.variables, src->second->getReadList(), dest->second->getReadList(), src->second->getCUDAStateListSize(), dest->second->getCUDAStateListSize());
            // Update list sizes
            dest->second->setCUDAStateListSize(dest->second->getCUDAStateListSize() + src->second->getCUDAStateListSize());
            src->second->setCUDAStateListSize(0);
        }
    }
}
void CUDAAgent::mapNewRuntimeVariables(const AgentFunctionData& func, const unsigned int &maxLen) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(func.agent_output_state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::mapNewRuntimeVariables()",
            agent_description.name.c_str(), func.agent_output_state.c_str());
    }

    const Curve::VariableHash _agent_birth_hash = Curve::getInstance().variableRuntimeHash("_agent_birth");
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        void* d_ptr = sm->second->getAgentNewListVariablePointer(mmp.first);

        // map using curve
        const Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());

        // get the agent variable size
        size_t size = mmp.second.type_size;

        // maximum population num
        Curve::getInstance().registerVariableByHash(var_hash + _agent_birth_hash + func_hash, d_ptr, size, maxLen);
    }
}

void CUDAAgent::unmapNewRuntimeVariables(const AgentFunctionData& func) const {
    const Curve::VariableHash _agent_birth_hash = Curve::getInstance().variableRuntimeHash("_agent_birth");
    const Curve::VariableHash func_hash = Curve::getInstance().variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        // void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // unmap using curve
        const Curve::VariableHash var_hash = Curve::getInstance().variableRuntimeHash(mmp.first.c_str());
        Curve::getInstance().unregisterVariableByHash(var_hash + _agent_birth_hash + func_hash);
    }
}
void CUDAAgent::resizeNew(const AgentFunctionData& func, const unsigned int &newSize, const unsigned int &streamId) {
    // Confirm agent output is set
    if (auto oa = func.agent_output.lock()) {
        // Resize new list of state_map
        auto sm = state_map.find(func.agent_output_state);
        if (sm != state_map.end()) {
            sm->second->resizeNewList(newSize);
        } else {
            THROW InvalidStateName("Agent '%s' does not contain state '%s', "
                "in CUDAAgent::resizeNew()\n",
                agent_description.name.c_str(), func.agent_output_state.c_str());
        }
        // Fill new list with default values
        sm->second->initNew(newSize, streamId);
        // Notify scan flag that it might need resizing
        // We need a 3rd array, because a function might combine agent birth, agent death and message output
        flamegpu_internal::CUDAScanCompaction::resize(newSize, flamegpu_internal::CUDAScanCompaction::AGENT_OUTPUT, streamId);
    }
}

void CUDAAgent::scatterNew(const std::string state, const unsigned int &newSize, const unsigned int &streamId) {
    auto sm = state_map.find(state);
    if (sm == state_map.end()) {
        THROW InvalidStateName("Agent '%s' does not contain state '%s', "
            "in CUDAAgent::scatterNew()\n",
            agent_description.name.c_str(), state.c_str());
    }
    // Perform scan
    if (newSize > flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].cub_temp_size_max_list_size) {
        if (flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].hd_cub_temp) {
            gpuErrchk(cudaFree(flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].hd_cub_temp));
        }
        flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].cub_temp_size = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].cub_temp_size,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].d_ptrs.scan_flag,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].d_ptrs.position,
            newSize + 1);
        gpuErrchk(cudaMalloc(&flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].hd_cub_temp,
            flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].cub_temp_size));
        flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].cub_temp_size_max_list_size = max_list_size;
    }
    cub::DeviceScan::ExclusiveSum(
        flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].hd_cub_temp,
        flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].cub_temp_size,
        flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].d_ptrs.scan_flag,
        flamegpu_internal::CUDAScanCompaction::hd_configs[flamegpu_internal::CUDAScanCompaction::Type::AGENT_OUTPUT][streamId].d_ptrs.position,
        newSize + 1);
    // Resize d_list if necessary
    if (sm->second->getCUDATrueStateListSize() + newSize > max_list_size) {
        resize(sm->second->getCUDATrueStateListSize() + newSize, streamId);
    }
    // Scatter
    if (newSize)
        sm->second->scatterNew(newSize, streamId);
}
