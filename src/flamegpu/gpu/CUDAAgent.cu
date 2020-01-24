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
    , max_list_size(0)
    , curve(Curve::getInstance()) { }

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
    // Notify scan flag this it might need resizing
    flamegpu_internal::CUDAScanCompaction::resizeAgents(max_list_size, streamId);
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
void CUDAAgent::mapRuntimeVariables(const AgentFunctionData& func) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(func.initial_state);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::mapRuntimeVariables()",
            agent_description.name.c_str(), func.initial_state.c_str());
    }

    const Curve::VariableHash agent_hash = curve.variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = curve.variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // map using curve
        const Curve::VariableHash var_hash = curve.variableRuntimeHash(mmp.first.c_str());

        // get the agent variable size
        size_t size = mmp.second.type_size;

       // maximum population num
        unsigned int length = this->getMaximumListSize();
        curve.registerVariableByHash(var_hash + agent_hash + func_hash, d_ptr, size, length);
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

    const Curve::VariableHash agent_hash = curve.variableRuntimeHash(agent_description.name.c_str());
    const Curve::VariableHash func_hash = curve.variableRuntimeHash(func.name.c_str());
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto &mmp : agent_description.variables) {
        // get a device pointer for the agent variable name
        // void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // unmap using curve
        const Curve::VariableHash var_hash = curve.variableRuntimeHash(mmp.first.c_str());
        curve.unregisterVariableByHash(var_hash + agent_hash + func_hash);
    }
}

void CUDAAgent::process_death(const AgentFunctionData& func, const unsigned int &streamId) {
    if (func.has_agent_death) {  // Optionally process agent death
        // check the cuda agent state map to find the correct state list for functions starting state
        CUDAStateMap::const_iterator sm = state_map.find(func.initial_state);

        unsigned int agent_count = sm->second->getCUDAStateListSize();
        if (agent_count > flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].cub_temp_size_max_list_size) {
            if (flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].hd_cub_temp) {
                gpuErrchk(cudaFree(flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].hd_cub_temp));
            }
            flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].cub_temp_size = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].cub_temp_size,
                flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.scan_flag,
                flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.position,
                max_list_size + 1);
            gpuErrchk(cudaMalloc(&flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].hd_cub_temp, flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].cub_temp_size));
            flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].cub_temp_size_max_list_size = max_list_size;
        }
        cub::DeviceScan::ExclusiveSum(
            flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].hd_cub_temp,
            flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].cub_temp_size,
            flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.scan_flag,
            flamegpu_internal::CUDAScanCompaction::hd_agent_configs[streamId].d_ptrs.position,
            agent_count + 1);
        // Scatter
        sm->second->scatter(streamId);
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
