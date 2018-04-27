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


//TODO: Some example code of the handle class and an example function

class MessageIterator;  // Forward declaration (class defined below)

class MessageIterator 
{

public:
    __device__ MessageIterator() {};

	using iterator = impl::MsgIterator;

	__device__ iterator begin(void);
    __device__ iterator end(void);
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
	* \param messagename_hash
	*/
	__device__ void setMessageOutpNameSpace(CurveNamespaceHash messagename_hash)
	{
		messagename_outp_hash = messagename_hash;
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
	CurveNamespaceHash agent_func_name_hash;
	CurveNamespaceHash messagename_inp_hash;
	CurveNamespaceHash messagename_outp_hash;

	unsigned int messageList_size;

};


__device__  iterator MessageIterator::begin()
{
	return;
}


__device__  iterator MessageIterator::end()
{
	return ;
}


__device__  auto MessageIterator::next()
{
	return 
}


/**
* \brief Gets an agent memory value
* \param variable_name Name of memory variable to retrieve
*/
template<typename T, unsigned int N>
__device__ T MessageIterator::getVariable(const char(&variable_name)[N])
{

	//simple indexing assumes index is the thread number (this may change later)
	unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;


	//get the value from curve
	T value = curveGetVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, index);

	//return the variable from curve
	return value;
}

#endif /* MESSAGE_ITERATOR_H_ */
