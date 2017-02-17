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
T void FLAMEGPU_API::getVariable(std::string name)
{
    int hash_index = this.getHashIndex(name);


}

template<typenae T>
T void FLAMEGPU_API::setVariable(std::string name, T value) {}


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
