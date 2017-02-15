/*
* flame_api.h
*
*  Created on: 19 Feb 2014
*      Author: paul
*/

#ifndef FLAME_FUNCTIONS_API_H_
#define FLAME_FUNCTIONS_API_H_
/*
singleton class FLAME
{
GetMessageIterator("messagename");
FLAME.getMem<type>("variablename");

};
*/

//TODO: Some example code of the handle class and an example function
//! FLAMEGPU_API is a singleton class
class FLAMEGPU_API;  // Forward declaration (class defined below)

//! Datatype for argument of all agent transition functions
typedef FLAMEGPU_API& FLAMEGPU_AgentFunctionParamType;
//! Datatype for return value for all agent transition functions
typedef FLAME_GPU_AGENT_STATUS FLAMEGPU_AgentFunctionReturnType;

// re-defined in AgentFunctionDescription
//! FLAMEGPU function return type
enum FLAME_GPU_AGENT_STATUS { ALIVE, DEAD };
//#define FLAMEGPU_AGENT_ALIVE 0
//#define FLAMEGPU_AGENT_DEAD  1

typedef std::map<const std::string, std::unique_ptr<FLAMEGPUStateList>> FGPU_StateMap;	//map of state name to CUDAAgentStateList which allocates memory on the device
typedef std::pair<const std::string, std::unique_ptr<FLAMEGPUStateList>> FGPU_StateMapPair;



//! Macro for defining agent transition functions with the correct input
//! argument and return type
#define FLAMEGPU_AGENT_FUNCTION(funcName) \
          FLAMEGPU_AgentFunctionReturnType \
          funcName(FLAMEGPU_AgentFunctionParamType FLAMEGPU)
/**
* @note Example Usage:

FLAMEGPU_AGENT_FUNCTION(move_func) {
  int x = FLAMEGPU.getVariable<int>("x");
  FLAMEGPU.setVariable<int>("x", x*10);
  return ALIVE;
}
*/


#define UNIFIED_GPU_MEMORY

/**
 * Stores a map of pointers to device memory locations. Must use the CUDAAgent hash table functions to access the correct index.
 */
struct CUDAAgentMemoryHashMap
{
    void **d_d_memory;	//device array of pointers to device variable arrays
    void **h_d_memory;  //host array of pointers to device variable arrays
};


/**
* Scott Meyers version of singleton class in his Effective C++ book
* Usage in main.cpp:
// Version 1
FLAMEGPU_API *p = &FLAMEGPU_API::getInstance(); // cache instance pointer
p->demo();
// Version 2
FLAMEGPU_API::getInstance().demo();
*
* @note advantages : The function-static object is initialized when the control flow is first passing its definition and no copy object creation is not allowed
* http://silviuardelean.ro/2012/06/05/few-singleton-approaches/
* @warning THIS IS NOT THE RIGHT WAY ! I should pass modeldescription object to this instead of agentDescription. In other words, the singleton class should iterate the agentmap and does what CUDAAgentModel do. (Under investigation)
*/
class FLAMEGPU_API
{
private:
    /* Here will be the instance stored. */
    static FLAMEGPU_API* instance;

    // Private constructor to prevent instancing.
    // Users can create object directly but with GetInstance() method.
    FLAMEGPU_API(const AgentDescription& description) : agent_description(description), state_map(), max_list_size(0); // or empty

        // copy constructor private
        FLAMEGPU_API(const FLAMEGPU_API& obj)
    {
        instance = obj.instance;
    }

    FLAMEGPU_API& operator = (const FLAMEGPU_API& rs)
    {
        if (this != &rs)
        {
            instance = rs.instance;
        }

        return *this;
    }
    //! private destructor, so users can't delete the pointer to the object by accident
    ~FLAMEGPU_API();

    const AgentDescription& agent_description;
    FGPU_StateMap state_map;

    unsigned int* h_hashes; //host hash index table //USE SHARED POINTER??
    unsigned int* d_hashes; //device hash index table (used by runtime)

    unsigned int max_list_size; //The maximum length of the agent variable arrays based on the maximum population size passed to setPopulationData


public:

    /**
    * static access method
    @return a reference instead of pointer
    */
    static FLAMEGPU_API* getInstance(const AgentDescription& description)
    {
        static FLAMEGPU_API theInstance(description);
        instance = &theInstance(description);

        // test :
        std::cout << "test in getInstance (Print Model Name): " << instance.agent_description.getName();
        return *instance;

    }


    void demo()
    {
        std::cout << "other singleton # next - your code ..." << std::endl;
    }

    unsigned int getHashListSize() const;

    int getHashIndex(const char * variable_name) const;

    const AgentDescription& getAgentDescription() const;

    /* Should be set initial population data and should only be called once as it does all the GPU device memory allocation */
    // void setInitialPopulationData(const AgentPopulation& population);

    /* Can be used to override the current population data without reallocating */
    //void setPopulationData(const AgentPopulation& population);

    //void getPopulationData(AgentPopulation& population);

    unsigned int getMaximumListSize() const;

    template<typename T>
    T getVariable(std::string name);

    template<typenae T>
    T setVariable(std::string name, T value);

protected:

    void zeroAllStateVariableData();
};

//! Initialize pointer. It's Null, because instance will be initialized on demand. */
FLAMEGPU_API* FLAMEGPU_API::instance = nullptr; //0

// usage: FLAMEGPU_API *p1 = FLAMEGPU_API::getInstance();
//FLAMEGPU_API* FLAMEGPU_API::getInstance()
//{
//    if (instance == 0) // is it the first call?
//    {
//        instance = new FLAMEGPU_API(); // create sole instance
//    }
//
//    return instance; // address of sole instance
//}



class CUDAAgentStateList
{
public:
    CUDAAgentStateList(FLAMEGPU_API& FGPU_agent);
    virtual ~CUDAAgentStateList();

    //cant be done in destructor as it requires access to the parent CUDAAgent object
    void cleanupAllocatedData();

    void setAgentData(const AgentStateMemory &state_memory);

    void getAgentData(AgentStateMemory &state_memory);

    void zeroAgentData();

protected:

    /*
     * The purpose of this function is to allocate on the device a block of memory for each variable. These vectors are stored within a hash list using the cuRVE technique so that the location of the vectors can be quickly determined at runtime by FLAME GPU functions.
     */
    void allocateDeviceAgentList(CUDAAgentMemoryHashMap* agent_list);

    void releaseDeviceAgentList(CUDAAgentMemoryHashMap* agent_list);

    void zeroDeviceAgentList(CUDAAgentMemoryHashMap* agent_list);


private:
    CUDAAgentMemoryHashMap d_list;
    CUDAAgentMemoryHashMap d_swap_list;
    CUDAAgentMemoryHashMap d_new_list;

    unsigned int current_list_size; //???

    FLAMEGPU_API& agent;
};

///////////////////////////////////////

/**
* FLAMEGPU_API class
* @brief allocates the hash table/list for agent variables and copy the list to device
*/
FLAMEGPU_API::FLAMEGPU_API(const AgentDescription& description) : agent_description(description), state_map(), max_list_size(0)
{

    //! allocate hash list
    h_hashes = (unsigned int*) malloc(sizeof(unsigned int)*getHashListSize());
    memset(h_hashes, EMPTY_HASH_VALUE, sizeof(unsigned int)*getHashListSize());
    gpuErrchk( cudaMalloc( (void**) &d_hashes, sizeof(unsigned int)*getHashListSize()));

    //! init hash list
    const MemoryMap &mem = agent_description.getMemoryMap();
    for (MemoryMap::const_iterator it = mem.begin(); it != mem.end(); it++)
    {

        //! save the variable hash in the host hash list
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
 * @brief Destroys the FLAMEGPU_API object
 */
FLAMEGPU_API::~FLAMEGPU_API(void)
{
    //loop through CUDAStateMap to clean up any cuda allocated data before the hash data is destroyed
    for (FGPU_StateMapPair &sm : state_map)
    {
        sm.second->cleanupAllocatedData();
    }
    free (h_hashes);
    gpuErrchk( cudaFree(d_hashes));
}


/**
* @brief Returns the hash list size
* @param none
* @return the hash list size required to store pointers to all agent variables (the size of hash table) that is double of the size of agent variables to minimise hash collisions
*/
unsigned int FLAMEGPU_API::getHashListSize() const
{
    return 2*agent_description.getNumberAgentVariables();
}

/**
* @brief Host function to get the hash index given a variable name. The hash list will have been completed by initialisation.
* @param agent variable name
* @return a hash index that corresponds to the agent variable name
*/
int FLAMEGPU_API::getHashIndex(const char * variable_name) const
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
const AgentDescription& FLAMEGPU_API::getAgentDescription() const
{
    return agent_description;
}

/**
* @brief Sets initial population data by allocating memory for each state list by creating a new agent state list
* @param AgentPopulation object
* @return none
*/
void FLAMEGPU_API::setInitialPopulationData(const AgentPopulation& population)
{
    //check that the initial population data has not already been set
    if (!state_map.empty())
        throw InvalidPopulationData("Error: Initial population data already set");

    //set the maximum population state size
    max_list_size = population.getMaximumStateListCapacity();

    //Make sure population uses same agent description as was used to initialise the agent FLAMEGPU_API
    if (&(population.getAgentDescription()) != &agent_description)
        throw InvalidPopulationData("Error: setInitialPopulationData population has a different agent description to that which was used to initialise the FLAMEGPU_API");

    //create map of device state lists by traversing the state list
    const StateMap& sm = agent_description.getStateMap();
    for(const StateMapPair& s: sm)
    {
        //allocate memory for each state list by creating a new Agent State List
        state_map.insert(FGPU_StateMap::value_type(s.first, std::unique_ptr<FLAMEGPU_APIStateList>( new FLAMEGPU_APIStateList(*this))));
    }

    /**set the population data*/
    setPopulationData(population);

}

/**
* @brief Sets the population data
* @param AgentPopulation object
* @return none
*/
void FLAMEGPU_API::setPopulationData(const AgentPopulation& population)
{
    //check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    if (state_map.empty())
        throw InvalidPopulationData("Error: Initial population data not set. Have you called setInitialPopulationData?");

    //check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() > max_list_size)
        throw InvalidPopulationData("Error: Maximum population size exceeds that of the initial population data?");

    //Make sure population uses same agent description as was used to initialise the agent FLAMEGPU_API
    const std::string agent_name = agent_description.getName();
    if (&(population.getAgentDescription()) != &agent_description)
        throw InvalidPopulationData("Error: setPopulationData population has a different agent description to that which was used to initialise the FLAMEGPU_API");


    /**set all population data to zero*/
    zeroAllStateVariableData();

    /**copy all population data to correct state map*/
    const StateMap& sm = agent_description.getStateMap();
    for (const StateMapPair& s : sm)
    {
        //get an associated CUDA statemap pair
        FGPU_StateMap::iterator i = state_map.find(s.first);

        /**check that the FLAMEGPU_APIStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end())

            throw InvalidMapEntry("Error: failed to find memory allocated for a state. This should never happen!");

        //copy the data from the population state memory to the state_maps FLAMEGPU_APIStateList
        i->second->setAgentData(population.getReadOnlyStateMemory(i->first));
    }

}



void FLAMEGPU_API::getPopulationData(AgentPopulation& population)
{
    //check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    if (state_map.empty())
        throw InvalidPopulationData("Error: Initial population data not set. Have you called setInitialPopulationData?");

    //check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() < max_list_size)
        throw InvalidPopulationData("Error: Maximum population size is not large enough for FLAMEGPU_API");

    //Make sure population uses same agent description as was used to initialise the agent FLAMEGPU_API
    const std::string agent_name = agent_description.getName();
    if (&(population.getAgentDescription()) != &agent_description)
        throw InvalidPopulationData("Error: getPopulationData population has a different agent description to that which was used to initialise the FLAMEGPU_API");


    /* copy all population from correct state maps */
    const StateMap& sm = agent_description.getStateMap();
    for (const StateMapPair& s : sm)
    {
        //get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s.first);

        /**check that the FLAMEGPU_APIStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end())

            throw InvalidMapEntry("Error: failed to find memory allocated for a state. This should never happen!");

        //copy the data from the population state memory to the state_maps FLAMEGPU_APIStateList
        i->second->getAgentData(population.getStateMemory(i->first));
    }

}

/**
* @brief Returns the maximum list size
* @param none
* @return maximum size list that is equal to the maxmimum population size
*/
unsigned int FLAMEGPU_API::getMaximumListSize() const
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
void FLAMEGPU_API::zeroAllStateVariableData()
{
    //loop through state maps and reset the values
    for (FGPU_StateMapPair& s : state_map)
    {
        s.second->zeroAgentData();
    }
}

//An example of how the getVariable should work
//1) Given the string name argument use runtime hashing to get a variable unsigned int (call function in RuntimeHashing.h)
//2) Call (a new local) getHashIndex function to check the actual index in the hash table for the variable name. Once found we have a pointer to the vector of data for that agent variable
//3) Using the CUDA thread and block index (threadIdx.x) return the specific agent variable value from the vector
// Useful existing code to look at is CUDAAgentStateList setAgentData function
// Note that this is using the hashing to get a specific pointer for a given variable name. This is exactly what we want to do in the FLAME GPU API class

template<typename T>
T void FLAMEGPU_API::getVariable(std::string name){}

template<typenae T>
T void FLAMEGPU_API::setVariable(std::string name, T value){}


/////////////////////////////////////////////////////////////// copied from CUDAAgentStateList class
/**
* CUDAAgentStateList class
* @brief populates CUDA agent map, CUDA message map
*/
CUDAAgentStateList::CUDAAgentStateList(FLAMEGPU_API& FGPU_agent) : agent(FGPU_agent)
{

    //allocate state lists
    allocateDeviceAgentList(&d_list);
    allocateDeviceAgentList(&d_swap_list);
    if (agent.getAgentDescription().requiresAgentCreation())
        allocateDeviceAgentList(&d_new_list);
    else
    {
        //set new list hash map pointers to zero
        d_new_list.d_d_memory = 0;
        d_new_list.h_d_memory = 0;
    }

}

/**
 * A destructor.
 * @brief Destroys the CUDAAgentStateList object
 */
CUDAAgentStateList::~CUDAAgentStateList()
{
    //if cleanupAllocatedData function has not been called throw an error.
    if (d_list.d_d_memory != 0)
    {
        throw InvalidOperation("Error cleaning up CUDAStateList data. cleanupAllocatedData function must be called!");
    }
}

void CUDAAgentStateList::cleanupAllocatedData()
{
    //clean up
    releaseDeviceAgentList(&d_list);
    d_list.d_d_memory = 0;
    d_list.h_d_memory = 0;
    releaseDeviceAgentList(&d_swap_list);
    d_swap_list.d_d_memory = 0;
    d_swap_list.h_d_memory = 0;
    if (agent.getAgentDescription().requiresAgentCreation())
    {
        releaseDeviceAgentList(&d_new_list);
        d_new_list.d_d_memory = 0;
        d_new_list.h_d_memory = 0;
    }

}

/**
* @brief Allocates Device agent list
* @param variable of type CUDAAgentMemoryHashMap struct type
* @return none
*/
void CUDAAgentStateList::allocateDeviceAgentList(CUDAAgentMemoryHashMap* memory_map)
{
    //we use the agents memory map to iterate the agent variables and do allocation within our GPU hash map
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

    //allocate host vector (the map) to hold device pointers
    memory_map->h_d_memory = (void**)malloc(sizeof(void*)*agent.getHashListSize());
    //set all map values to zero
    memset(memory_map->h_d_memory, 0, sizeof(void*)*agent.getHashListSize());

    //for each variable allocate a device array and register in the hash map
    for (const MemoryMapPair& mm : mem)
    {

        //get the hash index of the variable so we know what position to allocate in the map
        int hash_index = agent.getHashIndex(mm.first.c_str());

        //get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(mm.first);

        //do the device allocation at the correct index and store the pointer in the host hash map
        gpuErrchk(cudaMalloc((void**)&(memory_map->h_d_memory[hash_index]), var_size * agent.getMaximumListSize()));
    }

    //allocate device vector (the map) to hold device pointers (which have already been allocated)
    gpuErrchk(cudaMalloc((void**)&(memory_map->d_d_memory), sizeof(void*)*agent.getHashListSize()));

    //copy the host array of map pointers to the device array of map pointers
    gpuErrchk(cudaMemcpy(memory_map->d_d_memory, memory_map->h_d_memory, sizeof(void*)*agent.getHashListSize(), cudaMemcpyHostToDevice));



}

/**
* @brief Frees
* @param variable of type CUDAAgentMemoryHashMap struct type
* @return none
*/
void CUDAAgentStateList::releaseDeviceAgentList(CUDAAgentMemoryHashMap* memory_map)
{
    //we use the agents memory map to iterate the agent variables and do deallocation within our GPU hash map
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

    //for each device pointer in the map we need to free these
    for (const MemoryMapPair& mm : mem)
    {
        //get the hash index of the variable so we know what position to allocate
        int hash_index = agent.getHashIndex(mm.first.c_str());

        //free the memory on the device
        gpuErrchk(cudaFree(memory_map->h_d_memory[hash_index]));
    }

    //free the device memory map
    gpuErrchk(cudaFree(memory_map->d_d_memory));

    //free the host memory map
    free(memory_map->h_d_memory);
}

/**
* @brief
* @param variable of type CUDAAgentMemoryHashMap struct type
* @return none
*/
void CUDAAgentStateList::zeroDeviceAgentList(CUDAAgentMemoryHashMap* memory_map)
{
    //we use the agents memory map to iterate the agent variables and do deallocation within our GPU hash map
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();

    //for each device pointer in the map we need to free these
    for (const MemoryMapPair& mm : mem)
    {
        //get the hash index of the variable so we know what position to allocate
        int hash_index = agent.getHashIndex(mm.first.c_str());

        //get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(mm.first);

        //set the memory to zero
        gpuErrchk(cudaMemset(memory_map->h_d_memory[hash_index], 0, var_size*agent.getMaximumListSize()));
    }
}

/**
* @brief
* @param AgenstStateMemory object
* @return none
* @todo
*/
void CUDAAgentStateList::setAgentData(const AgentStateMemory &state_memory)
{

    //check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription()))
    {
        //throw std::runtime_error("CUDA Agent uses different agent description.");
        throw InvalidCudaAgentDesc();
    }


    //copy raw agent data to device pointers
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();
    for (const MemoryMapPair& m : mem)
    {
        //get the hash index of the variable so we know what position to allocate
        int hash_index = agent.getHashIndex(m.first.c_str());

        //get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(m.first);

        //get the vector
        const GenericMemoryVector &m_vec = state_memory.getReadOnlyMemoryVector(m.first);

        //get pointer to vector data
        const void * v_data = m_vec.getReadOnlyDataPtr();

        //set the current list size
        current_list_size = state_memory.getStateListSize();

        //TODO: copy the boost any data to GPU
        gpuErrchk(cudaMemcpy(d_list.h_d_memory[hash_index], v_data, var_size*current_list_size, cudaMemcpyHostToDevice));
    }

}

void CUDAAgentStateList::getAgentData(AgentStateMemory &state_memory)
{

    //check that we are using the same agent description
    if (!state_memory.isSameDescription(agent.getAgentDescription()))
    {
        //throw std::runtime_error("CUDA Agent uses different agent description.");
        throw InvalidCudaAgentDesc();
    }


    //copy raw agent data to device pointers
    const MemoryMap &mem = agent.getAgentDescription().getMemoryMap();
    for (const MemoryMapPair& m : mem)
    {
        //get the hash index of the variable so we know what position to allocate
        int hash_index = agent.getHashIndex(m.first.c_str());

        //get the variable size from agent description
        size_t var_size = agent.getAgentDescription().getAgentVariableSize(m.first);

        //get the vector
        GenericMemoryVector &m_vec = state_memory.getMemoryVector(m.first);

        //get pointer to vector data
        void * v_data = m_vec.getDataPtr();

        //check  the current list size
        if (current_list_size > state_memory.getPopulationCapacity())
            throw InvalidMemoryCapacity("Current GPU state list size exceed the state memory available!");

        //copy the GPU data to host
        gpuErrchk(cudaMemcpy(v_data, d_list.h_d_memory[hash_index], var_size*current_list_size, cudaMemcpyDeviceToHost));

        //set the new state list size
        state_memory.overrideStateListSize(current_list_size);
    }

}

/**
* @brief
* @param none
* @return none
*/
void CUDAAgentStateList::zeroAgentData()
{
    zeroDeviceAgentList(&d_list);
    zeroDeviceAgentList(&d_swap_list);
    if (agent.getAgentDescription().requiresAgentCreation())
        zeroDeviceAgentList(&d_new_list);
}




/**
//before this function would be called the GPU simulation engine would configure the FLAMEGPU_API object.
//This would map any variables to the hash table that the function has access to (but only these variables)

*
* @note Example Usage:
FLAME_GPU_AGENT_STATUS temp_func(FLAMEGPU_API *api){

	int x = api->getVariable<int>("x");
	x++;
	api->setVariable<int>("x", x);
}
*/


#endif /* FLAME_FUNCTIONS_API_H_ */
