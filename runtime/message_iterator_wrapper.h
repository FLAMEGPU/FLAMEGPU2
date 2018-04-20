#ifndef MESSAGE_ITERATOR_WRAPPER_H_
#define MESSAGE_ITERATOR_WRAPPER_H_

/**
 * @file message_iterator_wrapper.h
 * @author  
 * @date    
 * @brief  MessageIteratorWrapper is a wrapper class
 *
 * \todo longer description
 */


#include "../gpu/CUDAErrorChecking.h"			//required for CUDA error handling functions
#include "cuRVE/curve.h"
#include "../exception/FGPUException.h"


//TODO: Some example code of the handle class and an example function

class MessageIteratorWrapper;  // Forward declaration (class defined below)

class MessageIteratorWrapper
{

public:
    __device__ MessageIteratorWrapper() {};

  //  template<typename T, unsigned int N> __device__
   // T getVariable(const char(&variable_name)[N]);

	template <typename T> T getMessage(void);
	// we need to have get first message method, get next message method and possible get count 
	
	// getCount?
	// nextMessage?
	// firstMessage?
	// need a for loop to return based on the index and not the whole list
private:
	//CurveNamespaceHash agent_func_name_hash;

};


#endif /* MESSAGE_ITERATOR_WRAPPER_H_ */
