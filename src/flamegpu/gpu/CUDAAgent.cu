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
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/cuRVE/curve.h"

/**
* CUDAAgent class
* @brief allocates the hash table/list for agent variables and copy the list to device
*/
CUDAAgent::CUDAAgent(const AgentDescription& description) : agent_description(description), state_map(), max_list_size(0) {
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
const AgentDescription& CUDAAgent::getAgentDescription() const {
    return agent_description;
}

/**
* @brief Sets initial population data by allocating memory for each state list by creating a new agent state list
* @param AgentPopulation object
* @return none
*/
void CUDAAgent::setInitialPopulationData(const AgentPopulation& population) {
    // check that the initial population data has not already been set
    if (!state_map.empty()) {
        THROW InvalidPopulationData("Error: Initial population data for agent '%s' already set. "
            "In CUDAAgent::setInitialPopulationData()",
            population.getAgentName().c_str());
    }
    // set the maximum population state size
    max_list_size = population.getMaximumStateListCapacity();

    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    if (&(population.getAgentDescription()) != &agent_description) {
        THROW InvalidPopulationData("Error: Initial Population has a different agent description ('%s') "
            "to that which was used to initialise the CUDAAgent ('%s'). "
            "In CUDAAgent::setInitialPopulationData()",
            population.getAgentName().c_str(),
            agent_description.getName().c_str());
    }
    // create map of device state lists by traversing the state list
    const StateMap& sm = agent_description.getStateMap();
    for (const StateMapPair& s : sm) {
        // allocate memory for each state list by creating a new Agent State List
        state_map.insert(CUDAStateMap::value_type(s.first, std::unique_ptr<CUDAAgentStateList>( new CUDAAgentStateList(*this))));
    }

    /**set the population data*/
    setPopulationData(population);
}

/**
* @brief Sets the population data
* @param AgentPopulation object
* @return none
*/
void CUDAAgent::setPopulationData(const AgentPopulation& population) {
    // check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    if (state_map.empty()) {
        THROW InvalidPopulationData("Error: Initial population data for agent '%s' not allocated. "
            "Have you called setInitialPopulationData()? "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str());
    }
    // check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() > max_list_size) {
        THROW InvalidPopulationData("Error: Maximum population size for agent '%s' exceeds allocation. "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str());
    }
    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    const std::string agent_name = agent_description.getName();
    if (&(population.getAgentDescription()) != &agent_description) {
        THROW InvalidPopulationData("Error: Initial Population has a different agent description ('%s') "
            "to that which was used to initialise the CUDAAgent ('%s'). "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str(),
            agent_description.getName().c_str());
    }
    /**set all population data to zero*/
    zeroAllStateVariableData();

    /**copy all population data to correct state map*/
    const StateMap& sm = agent_description.getStateMap();
    for (const StateMapPair& s : sm) {
        // get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s.first);

        /**check that the CUDAAgentStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end()) {
            THROW InvalidMapEntry("Error: failed to find memory allocated for agent ('%s') state ('%s') "
                "In CUDAAgent::setPopulationData() ",
                "This should never happen!",
                population.getAgentName().c_str(), s.first.c_str());
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
    // check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() < max_list_size) {
        THROW InvalidPopulationData("Error: Maximum population size for agent '%s' exceeds allocation. "
            "In CUDAAgent::getPopulationData()",
            population.getAgentName().c_str());
    }
    // Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    const std::string agent_name = agent_description.getName();
    if (&(population.getAgentDescription()) != &agent_description) {
        THROW InvalidPopulationData("Error: Initial Population has a different agent description ('%s') "
            "to that which was used to initialise the CUDAAgent ('%s'). "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str(),
            agent_description.getName().c_str());
    }
    /* copy all population from correct state maps */
    const StateMap& sm = agent_description.getStateMap();
    for (const StateMapPair& s : sm) {
        // get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s.first);

        /**check that the CUDAAgentStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end()) {
            THROW InvalidMapEntry("Error: failed to find memory allocated for agent ('%s') state ('%s') "
                "In CUDAAgent::setPopulationData() ",
                "This should never happen!",
                population.getAgentName().c_str(), s.first.c_str());
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
void CUDAAgent::mapRuntimeVariables(const AgentFunctionDescription& func) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(func.getInitialState());

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::mapRuntimeVariables()",
            agent_description.getName().c_str(), func.getInitialState().c_str());
    }

    // loop through the agents variables to map each variable name using cuRVE
    for (MemoryMapPair mmp : agent_description.getMemoryMap()) {
        // get a device pointer for the agent variable name
        void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // map using curve
        CurveVariableHash var_hash = curveVariableRuntimeHash(mmp.first.c_str());
        CurveVariableHash agent_hash = curveVariableRuntimeHash(func.getParent().getName().c_str());
        CurveVariableHash func_hash = curveVariableRuntimeHash(func.getName().c_str());

        // get the agent variable size
        size_t size;
        size = agent_description.getAgentVariableSize(mmp.first.c_str());

       // maximum population num
        unsigned int length = this->getMaximumListSize();

        curveRegisterVariableByHash(var_hash + agent_hash + func_hash, d_ptr, size, length);
    }
}

void CUDAAgent::unmapRuntimeVariables(const AgentFunctionDescription& func) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(func.getInitialState());

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::unmapRuntimeVariables()",
            agent_description.getName().c_str(), func.getInitialState().c_str());
    }

    // loop through the agents variables to map each variable name using cuRVE
    for (MemoryMapPair mmp : agent_description.getMemoryMap()) {
        // get a device pointer for the agent variable name
        // void* d_ptr = sm->second->getAgentListVariablePointer(mmp.first);

        // unmap using curve
        CurveVariableHash var_hash = curveVariableRuntimeHash(mmp.first.c_str());
        CurveVariableHash agent_hash = curveVariableRuntimeHash(func.getParent().getName().c_str());
        CurveVariableHash func_hash = curveVariableRuntimeHash(func.getName().c_str());

        curveUnregisterVariableByHash(var_hash + agent_hash + func_hash);
    }
}


const std::unique_ptr<CUDAAgentStateList> &CUDAAgent::getAgentStateList(const std::string &state_name) const {
    // check the cuda agent state map to find the correct state list for functions starting state
    CUDAStateMap::const_iterator sm = state_map.find(state_name);

    if (sm == state_map.end()) {
        THROW InvalidCudaAgentState("Error: Agent ('%s') state ('%s') was not found "
            "in CUDAAgent::getAgentStateList()",
            agent_description.getName().c_str(), state_name.c_str());
    }
    return sm->second;
}
