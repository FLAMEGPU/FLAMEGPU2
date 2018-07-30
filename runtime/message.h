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
//#include "messagelist.h"

//TODO: Some example code of the handle class and an example function

class Message;  // Forward declaration (class defined below)
class MessageList;

//
//class Message {
//
//public:
//	__device__ Message() : index(0) {};
//	__device__ Message(unsigned int index) : index(index) {};
//	//template<typename T, unsigned int N> __device__ T getVariable(MessageIterator mi, const char(&variable_name)[N]);
//	__host__ __device__ bool operator==(const Message& rhs) { return  index == rhs.index; }
//	__host__ __device__ bool operator!=(const Message& rhs) { return  index != rhs.index; }
//	__host__ __device__ Message& operator++() { ++index;  return *this; }
//	unsigned int index; // @todo - make this private/protected? We don't want the end user accessing it. Or make the variable private and have a public/protected getter so we can access it in the Iterator/ MessageList classes
//	template<typename T, unsigned int N>
//	__device__ T getVariable(MessageList messageList, const char(&variable_name)[N]);
//private:
//
//
//};
//
//template<typename T, unsigned int N>
//__device__ T Message::getVariable(MessageList messageList, const char(&variable_name)[N]) {
//	T value = curveGetVariable<T>(variable_name, messageList.getAgentNameSpace() + messageList.getMessageInpNameSpace(), this->index);
//	return value;
//}

#endif /* MESSAGE_H_ */
