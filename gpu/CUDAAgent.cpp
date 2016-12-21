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

#include "CUDAAgent.h"
#include "CUDAAgentStateList.h"
#include "RuntimeHashing.h"
#include "CUDAErrorChecking.h"

#include "../model/AgentDescription.h"
#include "../pop/AgentPopulation.h"


/**
* CUDAAgent class
* @brief allocates the hash table/list for agent variables and copy the list to device
*/
CUDAAgent::CUDAAgent(const AgentDescription& description) : agent_description(description), state_map(), max_list_size(0)
{

    //allocate hash list
    h_hashes = (unsigned int*) malloc(sizeof(unsigned int)*getHashListSize());
    memset(h_hashes, EMPTY_HASH_VALUE, sizeof(unsigned int)*getHashListSize());
    gpuErrchk( cudaMalloc( (void**) &d_hashes, sizeof(unsigned int)*getHashListSize()));

    //init hash list
    const MemoryMap &mem = agent_description.getMemoryMap();
    for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++)
    {

        //save the variable hash in the host hash list
        unsigned int hash = VariableHash(it->first.c_str());
        unsigned int n = 0;
        unsigned int i = (hash) % getHashListSize();	// agent_description.getNumberAgentVariables();

        while (h_hashes[i] != EMPTY_HASH_VALUE)
        {
            n += 1;
            if (n >= getHashListSize())
            {
                //throw std::runtime_error("Hash list full. This should never happen.");
                throw InvalidHashList();
            }
            i += 1;
            if (i >= getHashListSize())
            {
                i = 0;
            }
        }
        h_hashes[i] = hash;
    }

    //copy hash list to device
    gpuErrchk( cudaMemcpy( d_hashes, h_hashes, sizeof(unsigned int) * getHashListSize(), cudaMemcpyHostToDevice));

}


/**
 * A destructor.
 * @brief Destroys the CUDAAgent object
 */
CUDAAgent::~CUDAAgent(void)
{
    free (h_hashes);
    gpuErrchk( cudaFree(d_hashes));
}



/**
* @brief Returns the hash list size
* @param none
* @return the size of hash table that is double of the size of agent variables
*/
unsigned int CUDAAgent::getHashListSize() const
{
    return 2*agent_description.getNumberAgentVariables();
}

/**
* @brief Returns hash index
* @param agent variable name
* @return a hash index that corresponds to the agent variable name
*/
int CUDAAgent::getHashIndex(const char * variable_name) const
{
    //function resolves hash collisions
    unsigned int hash = VariableHash(variable_name);
    unsigned int n = 0;
	unsigned int i = (hash) % getHashListSize();

    while (h_hashes[i] != EMPTY_HASH_VALUE)
    {
        if (h_hashes[i] == hash)
        {
            return i; //return index if hash is found
        }
        n += 1;
        if (n >= getHashListSize())
        {
            //throw std::runtime_error("Hash list full. This should never happen.");
            throw InvalidHashList();
        }
        i += 1;
        if (i >= getHashListSize())
        {
            i = 0;
        }
    }

    //throw std::runtime_error("This should be an unknown variable error. Should never occur");
    throw InvalidAgentVar("This should be an unknown variable error. Should never occur");
    return -1; //return invalid index
}

/**
* @brief Returns agent description
* @param none
* @return AgentDescription object
*/
const AgentDescription& CUDAAgent::getAgentDescription() const
{
    return agent_description;
}

/**
* @brief Sets initial population data by allocating memory for each state list by creating a new agent state list
* @param AgentPopulation object
* @return none
*/
void CUDAAgent::setInitialPopulationData(const AgentPopulation& population)
{
    //check that the initial population data has not already been set
    if (!state_map.empty())
        throw InvalidPopulationData("Error: Initial population data already set");

    //set the maximum population state size
    max_list_size = population.getMaximumStateListCapacity();

    //Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    if (&(population.getAgentDescription()) != &agent_description)
        throw InvalidPopulationData("Error: setInitialPopulationData population has a different agent description to that which was used to initialise the CUDAAgent");

    //create map of device state lists by traversing the state list
    const StateMap& sm = agent_description.getStateMap();
    for(const StateMapPair& s: sm)
    {
        //allocate memory for each state list by creating a new Agent State List
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
void CUDAAgent::setPopulationData(const AgentPopulation& population)
{
    //check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    if (state_map.empty())
        throw InvalidPopulationData("Error: Initial population data not set. Have you called setInitialPopulationData?");

    //check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() > max_list_size)
        throw InvalidPopulationData("Error: Maximum population size exceeds that of the initial population data?");

    //Make sure population uses same agent description as was used to initialise the agent CUDAAgent
    const std::string agent_name = agent_description.getName();
    if (&(population.getAgentDescription()) != &agent_description)
        throw InvalidPopulationData("Error: setPopulationData population has a different agent description to that which was used to initialise the CUDAAgent");


    /**set all population data to zero*/
    zeroAllStateVariableData();

    /**copy all population data to correct state map*/
    const StateMap& sm = agent_description.getStateMap();
    for (const StateMapPair& s : sm)
    {
        //get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s.first);

        /**check that the CUDAAgentStateList was found (should ALWAYS be the case)*/
		if (i == state_map.end())
			//throw std::exception("Error: failed to find memory allocated for a state. This should never happen!");
			throw InvalidMapEntry("Error: failed to find memory allocated for a state. This should never happen!");

        //copy the data from the population state memory to the state_maps CUDAAgentStateList
        i->second->setAgentData(population.getStateMemory(i->first));
    }

}

/**
* @brief Returns the maximum list size
* @param none
* @return maximum size list that is equal to the maxmimum population size
*/
unsigned int CUDAAgent::getMaximumListSize() const
{
    return max_list_size;
}


/**
* @brief Sets all state variable data to zero
* It loops through sate maps and resets the values
* @param none
* @return none
* @warning zeroAgentData
* @todo 'zeroAgentData'
*/
void CUDAAgent::zeroAllStateVariableData()
{
    //loop through state maps and reset the values
    for (CUDAStateMapPair& s : state_map)
    {
        s.second->zeroAgentData();
    }
}
