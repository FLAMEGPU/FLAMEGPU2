#ifndef MESSAGE_H_
#define MESSAGE_H_

/**
 * @file message.h
 * @author  
 * @date    
 * @brief  Message  class
 *
 * \todo longer description
 */

#include "../exception/FGPUException.h"

//TODO: Some example code of the handle class and an example function
class MessageList;

class Message;  // Forward declaration (class defined below)

class Message{

public:
    __device__ Message() : index(0){};
	__device__ Message(unsigned int index): index(index) {};
	template<typename T, unsigned int N> __device__ T getVariable(MessageList ml, const char(&variable_name)[N]);
	__host__ __device__ bool operator==(const Message& rhs) { return  index == rhs.index; }
	__host__ __device__ bool operator!=(const Message& rhs) { return  index != rhs.index; }
	__host__ __device__ Message& operator++() { ++index;  return *this; }
	unsigned int index; // @todo - make this private/protected? We don't want the end user accessing it. Or make the variable private and have a public/protected getter so we can access it in the Iterator/ MessageList classes
private:
	

};

/*
template<typename T, unsigned int N> 
__device__ T Message::getVariable(MessageList ml, const char(&variable_name)[N])
{
	return ml.getVariable<T>(variable_name);
}
*/

#endif /* MESSAGE_H_ */
