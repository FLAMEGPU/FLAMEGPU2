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

//singleton class (only one of them)
//class FLAMEGPU_API{
//    public:
//    FLAMEGPU_API():
//	//use the has list to get the right piece of memory
//	template<typenae T>
//	T getVariable(std::string name)
//	{
//		agent->getVariable<T>(agent.getName(), "x")
//	}
//
//		template<typenae T>
//	T setVariable(std::string name, T value)
//	{
//		agent->setVariable<T>(name, value)
//	}
//
//private:
//	//data structures for hash list
//};



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
