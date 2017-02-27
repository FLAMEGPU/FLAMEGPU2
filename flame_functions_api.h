/*
* flame_api.h
*
*  Created on: 19 Feb 2014
*      Author: paul
*/

#ifndef FLAME_FUNCTIONS_API_H_
#define FLAME_FUNCTIONS_API_H_

#include "gpu/CUDAErrorChecking.h"			//required for CUDA error handling functions
#include "runtime/cuRVE/curve.h"


//TODO: Some example code of the handle class and an example function
//! FLAMEGPU_API is a singleton class
class FLAMEGPU_API;  // Forward declaration (class defined below)

//! FLAMEGPU function return type
enum FLAME_GPU_AGENT_STATUS { ALIVE, DEAD };

/**
 * @brief FLAMEGPU function pointer definition
 */
typedef FLAME_GPU_AGENT_STATUS(*FLAMEGPU_AGENT_FUNCTION_POINTER)(FLAMEGPU_API *api);

/**
 * Macro for defining agent transition functions with the correct input. Must always be a device function to be called by CUDA.
 *
 * FLAMEGPU_AGENT_FUNCTION(move_func) {
 *   int x = FLAMEGPU.getVariable<int>("x");
 *   FLAMEGPU.setVariable<int>("x", x*10);
 * return ALIVE;
 *}
 */
#define FLAMEGPU_AGENT_FUNCTION(funcName) FLAME_GPU_AGENT_STATUS funcName(FLAMEGPU_API* FLAMEGPU)



/** @brief	A flame gpu api class for the device runtime only
 *
 * This class should only be used by the device and never created on the host. It is safe for each agent function to create a copy of this class on the device. Any singleton type
 * behaviour is handled by the curveInstance class. This will ensure that initialisation of the curve (C) library is done only once.
 */
class FLAMEGPU_API
{

public:
	__device__ FLAMEGPU_API(){};

	template<typename T, unsigned int N> __device__
		T getVariable(const char(&variable_name)[N]);

	template<typename T, unsigned int N> __device__
		void setVariable(const char(&variable_name)[N], T value);

};


/******************************************************************************************************* Implementation ********************************************************/

//An example of how the getVariable should work
//1) Given the string name argument use runtime hashing to get a variable unsigned int (call function in RuntimeHashing.h)
//2) Call (a new local) getHashIndex function to check the actual index in the hash table for the variable name. Once found we have a pointer to the vector of data for that agent variable
//3) Using the CUDA thread and block index (threadIdx.x) return the specific agent variable value from the vector
// Useful existing code to look at is CUDAAgentStateList setAgentData function
// Note that this is using the hashing to get a specific pointer for a given variable name. This is exactly what we want to do in the FLAME GPU API class

template<typename T, unsigned int N>
__device__ T FLAMEGPU_API::getVariable(const char(&variable_name)[N])
{
	//simple indexing assumes index is the thread number (this may change later)
	unsigned int index =  (blockDim.x * blockIdx.x) + threadIdx.x;

	//get the value from curve
	T value = curveGetVariable<T>(variable_name, index);

	//TODO: Some error checking?

	//return the variable from curve
	return value;
}

template<typename T, unsigned int N>
__device__ void FLAMEGPU_API::setVariable(const char(&variable_name)[N], T value){

	//simple indexing assumes index is the thread number (this may change later)
	unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	//set the variable using curve
	curveSetVariable<T>(variable_name, value, index);
}

#endif /* FLAME_FUNCTIONS_API_H_ */
