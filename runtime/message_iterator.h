#ifndef MESSAGE_ITERATOR_H_
#define MESSAGE_ITERATOR_H_

/**
 * @file message_iterator.h
 * @author  
 * @date    
 * @brief  MessageIterator is a wrapper class
 *
 * \todo longer description
 */


#include "../gpu/CUDAErrorChecking.h"			//required for CUDA error handling functions
#include "cuRVE/curve.h"
#include "../exception/FGPUException.h"
#include "custom_iter.h"


//TODO: Some example code of the handle class and an example function

class MessageIterator;  // Forward declaration (class defined below)


class MessageIterator 
{

public:
   // __device__ MessageIterator() {};
	__device__ MessageIterator(unsigned int start= 0, unsigned int end = 0) : start_(start), messageList_size(end) {};

	__device__ ~MessageIterator() {};

	__device__ custom_iter begin() { return custom_iter(start_); }
	__device__ custom_iter end() { return custom_iter(messageList_size); }

	//__device__ auto next(void);

	template<typename T, unsigned int N> __device__
		T getVariable(const char(&variable_name)[N]);

	/**
	* \brief
	* \param agentname_hash
	*/
	__device__ void setAgentNameSpace(CurveNamespaceHash agentname_hash)
	{
		agent_func_name_hash = agentname_hash;
	}

	/**
	* \brief
	* \param messagename_hash
	*/
	__device__ void setMessageInpNameSpace(CurveNamespaceHash messagename_hash)
	{
		messagename_inp_hash = messagename_hash;
	}

	/**
	* \brief
	* \param  messageList_Size
	*/
	__device__ void setMessageListSize(unsigned int message_size)
	{
		messageList_size = message_size;
	}

private:
	const unsigned int start_;
	//const unsigned  end_;
	unsigned int messageList_size;

	int index = 0;
	CurveNamespaceHash agent_func_name_hash;
	CurveNamespaceHash messagename_inp_hash;

};

/**
* \brief Gets an agent memory value
* \param variable_name Name of memory variable to retrieve
*/
template<typename T, unsigned int N>
__device__ T MessageIterator::getVariable(const char(&variable_name)[N])
{

	//get the value from curve
	T value = curveGetVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, index);

	index++;

	//return the variable from curve
	return value;
}

#endif /* MESSAGE_ITERATOR_H_ */
