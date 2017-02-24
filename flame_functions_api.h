/*
* flame_api.h
*
*  Created on: 19 Feb 2014
*      Author: paul
*/

#ifndef FLAME_FUNCTIONS_API_H_
#define FLAME_FUNCTIONS_API_H_


#include "gpu/RuntimeHashing.h"				//required for runtime hashing of strings
#include "gpu/CUDAErrorChecking.h"			//required for CUDA error handling functions
#include "gpu/cuRVE/curve.h"


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
* @warning THIS IS NOT THE RIGHT WAY ! I should pass model description object to this instead of agentDescription. In other words, the singleton class should iterate the agentmap and does what CUDAAgentModel do. (Under investigation)
* Paul: The singleton class should not have ANY C++ objects at all. Its should be configured by objects but kept its functions should simple as possible (i.e. using only the hash tables) to reduce overhead in the CUDA runtime.
* Paul: IMORTANT: We should migrate all hashing stuff to curve. This will simplify our own code lots. Some changes to curve are required. I have started to do these.
*/
class FLAMEGPU_API
{
private:
    // Private constructor to prevent instancing. Users can create object directly but with GetInstance() method.
	FLAMEGPU_API();

	//! private destructor, so users can't delete the pointer to the object by accident
	~FLAMEGPU_API();

    // copy constructor private
	FLAMEGPU_API(const FLAMEGPU_API& obj);

	//overload = operator (@question Mozhgan why is this required?)
	FLAMEGPU_API& operator= (const FLAMEGPU_API& rs);

public:

    /**
    * static access method
    @return a reference instead of pointer
    */
	static FLAMEGPU_API* getInstance();


    template<typename T>
    T getVariable(std::string variable_name);

    template<typename T>
    T setVariable(std::string variable_name, T value);

private:
	
	static FLAMEGPU_API* instance;	/* Here will be the instance stored. */

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

/******************************************************************************************************* Implementation ********************************************************/
/**
* FLAMEGPU_API class
* @brief allocates the hash table/list for agent variables and copy the list to device
*/
FLAMEGPU_API::FLAMEGPU_API()
{

	//init curve for runtime hashing of variables
	curveInit();
}

//copy constructor implementation
FLAMEGPU_API::FLAMEGPU_API(const FLAMEGPU_API& obj)
{
	instance = obj.instance;
}

FLAMEGPU_API& FLAMEGPU_API::operator= (const FLAMEGPU_API& rs)
{
	if (this != &rs)
	{
		instance = rs.instance;
	}

	return *this;
}

static FLAMEGPU_API* FLAMEGPU_API::getInstance()
{
	static FLAMEGPU_API theInstance();
	instance = &theInstance();

	return *instance;

}



/**
 * A destructor.
 * @brief Destroys the FLAMEGPU_API object
 */
FLAMEGPU_API::~FLAMEGPU_API(void)
{
	//curve requires no clean up as all memory is statically allocated at compile time (required for CUDA const memory)
}






//An example of how the getVariable should work 
//1) Given the string name argument use runtime hashing to get a variable unsigned int (call function in RuntimeHashing.h)
//2) Call (a new local) getHashIndex function to check the actual index in the hash table for the variable name. Once found we have a pointer to the vector of data for that agent variable
//3) Using the CUDA thread and block index (threadIdx.x) return the specific agent variable value from the vector
// Useful existing code to look at is CUDAAgentStateList setAgentData function
// Note that this is using the hashing to get a specific pointer for a given variable name. This is exactly what we want to do in the FLAME GPU API class

template<typename T, unisgned int N>
T __device__ void FLAMEGPU_API::getVariable(const char(&variable_name)[N])
{
	//simple indexing assumes index is the thread number (this may change later)
	unsigned int index =  (blockDim.x * blockIdx.x) + threadIdx.x;

	//get the value from curve
	T value = curveGetVariable<T>(variable_name, index);

	//TODO: Some error checking?

	//return the variable from curve
	return value;
}

template<typename T, unisgned int N>
T __device__ void FLAMEGPU_API::setVariable(const char(&variable_name)[N], T value){

	//simple indexing assumes index is the thread number (this may change later)
	unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	//set the variable using curve
	curveSetVariable<T>(variable_name, value, index);
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
