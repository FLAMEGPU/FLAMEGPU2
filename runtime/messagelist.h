#ifndef MESSAGELIST_H_
#define MESSAGELIST_H_

/**
 * @file messagelist.h
 * @author  
 * @date    
 * @brief  MessageList is a wrapper class
 *
 * \todo longer description
 */


#include "../gpu/CUDAErrorChecking.h"			//required for CUDA error handling functions
#include "cuRVE/curve.h"
#include "iterator"
#include "../exception/FGPUException.h"
#include "message.h"


using namespace std;

//TODO: Some example code of the handle class and an example function

class MessageList;  // Forward declaration (class defined below)
class Message;

class Message {

public:
	__device__ Message() : index(0) {};
	__device__ Message(unsigned int index) : index(index) {};
	//template<typename T, unsigned int N> __device__ T getVariable(MessageIterator mi, const char(&variable_name)[N]);
	__host__ __device__ bool operator==(const Message& rhs) { return  index == rhs.index; }
	__host__ __device__ bool operator!=(const Message& rhs) { return  index != rhs.index; }
	__host__ __device__ Message& operator++() { ++index;  return *this; }
	unsigned int index; // @todo - make this private/protected? We don't want the end user accessing it. Or make the variable private and have a public/protected getter so we can access it in the Iterator/ MessageList classes
	template<typename T, unsigned int N>
	__device__ T getVariable(MessageList messageList, const char(&variable_name)[N]);
private:


};


class MessageList 
{

public:

	class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void>
	{
		Message _message;
		MessageList &_parent;

	public:
		__host__ __device__ iterator(MessageList &parent, unsigned int index) : _parent(parent), _message(index) {}
		//iterator(MessageList* vector) : _message(vector) {}
		__host__ __device__ iterator& operator++() { ++_message;  return *this; }
		__host__ __device__ iterator operator++(int) { iterator tmp(*this); operator++(); return tmp; }
		__host__ __device__ bool operator==(const iterator& rhs) { return  _message == rhs._message; }
		__host__ __device__ bool operator!=(const iterator& rhs) { return  _message != rhs._message; }
		__host__ __device__ Message& operator*() { return _message; }
		template<typename T, unsigned int N> __device__
			T getVariable(const char(&variable_name)[N]);
	};

    //__device__ MessageList() {};
	__device__ MessageList(unsigned int start= 0, unsigned int end = 0) : start_(start), messageList_size(end) {};

	__device__ ~MessageList() {};

	//__device__ custom_iter begin() { return custom_iter(start_); }
	//__device__ custom_iter end() { return custom_iter(messageList_size); }

	/*! Returns the number of elements in the message list.
	*/
	inline __host__ __device__ std::size_t size(void) const //size_type
	{
		return messageList_size;
	}

	inline __host__ __device__ iterator begin(void) //const
	{
		return iterator(*this, start_);
	}
	inline __host__ __device__ iterator end(void) //const
	{
		return iterator(*this, start_ + size());
	}

	//template typename<T> T getVariable(char* name, MessageList:iterator it);
	// Possible provide another version which takes a Message too?

	template<typename T, unsigned int N> __device__
		T getVariable(MessageList::iterator iterator, const char(&variable_name)[N]);
	template<typename T, unsigned int N> __device__
		T getVariable(Message message, const char(&variable_name)[N]);

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


	__device__ CurveNamespaceHash getAgentNameSpace() {
		return this->agent_func_name_hash;
	}

	__device__ CurveNamespaceHash getMessageInpNameSpace() {
		return this->messagename_inp_hash;
	}

private:

	unsigned int start_;
	unsigned int end_;

	std::size_t messageList_size;

	// @todo - why does this have a pointer to itself?
	MessageList *message_;

	CurveNamespaceHash agent_func_name_hash;
	CurveNamespaceHash messagename_inp_hash;


};



//template<typename T, unsigned int N> 
//__device__ T Message::getVariable(MessageList messageList, const char(&variable_name)[N])
//{
//	return mi.getVariable<T>(variable_name);
//}


/**
* \brief Gets an agent memory value
* \param variable_name Name of memory variable to retrieve
*/
template<typename T, unsigned int N>
__device__ T MessageList::getVariable(MessageList::iterator iterator, const char(&variable_name)[N])
{

	//get the value from curve
	//T value = curveGetVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, (*iterator).index);
	T value = iterator.getVariable<T>(variable_name);

	//return the variable from curve
	return value;
}

template<typename T, unsigned int N>
__device__ T MessageList::getVariable(Message message, const char(&variable_name)[N])
{

	//get the value from curve
	T value = curveGetVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, message.index);

	//return the variable from curve
	return value;
}

template<typename T, unsigned int N>
__device__ T MessageList::iterator::getVariable(const char(&variable_name)[N])
{

	//get the value from curve
	//T value = curveGetVariable<T>(variable_name, _parent.agent_func_name_hash + _parent.messagename_inp_hash, this->_message.index);

	T value = this->_message.getVariable<T>(this->_parent, variable_name);
	//return the variable from curve
	return value;
}

template<typename T, unsigned int N>
__device__ T Message::getVariable(MessageList messageList, const char(&variable_name)[N]) {
	T value = curveGetVariable<T>(variable_name, messageList.getAgentNameSpace() + messageList.getMessageInpNameSpace(), this->index);
	return value;
}




#endif /* MESSAGELIST_H_ */
