#ifndef MESSAGES_H_
#define MESSAGES_H_

/**
 * @file messages.h
 * @author  
 * @date    
 * @brief  Messages  class
 *
 * \todo longer description
 */


#include "../exception/FGPUException.h"
#include "message_iterator.h"


//TODO: Some example code of the handle class and an example function

class Message;  // Forward declaration (class defined below)

class Message{

public:
    __device__ Message() {};

	template<typename T, unsigned int N> __device__
		T getVariable(MessageIterator mi, const char(&variable_name)[N]);

private:

};

template<typename T, unsigned int N> 
__device__ T Message::getVariable(MessageIterator mi, const char(&variable_name)[N])
{
	return mi.getVariable<T>(variable_name);
}


#endif /* MESSAGES_H_ */
