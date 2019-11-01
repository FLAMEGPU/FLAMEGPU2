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

#include <iterator>
#include <flamegpu/gpu/CUDAErrorChecking.h>            // required for CUDA error handling functions
#include "cuRVE/curve.h" // @todo migrate
#include <flamegpu/exception/FGPUException.h>

// TODO: Some example code of the handle class and an example function

class MessageList;  // Forward declaration (class defined below)

class MessageList  {


 public:

    typedef unsigned int size_type;

    class Message; // Forward declare inner classes
    class iterator; // Forward declare inner classes

    // Inner class representing an individual message
    class Message {

    private:
        MessageList &_messageList;
        size_type index;
    public:
    __device__ Message(MessageList &messageList) : _messageList(messageList), index(0) {};
    __device__ Message(MessageList &messageList, size_type index) : _messageList(messageList), index(index) {};
    __host__ __device__ bool operator==(const Message& rhs) { return  this->getIndex() == rhs.getIndex(); }
    __host__ __device__ bool operator!=(const Message& rhs) { return  this->getIndex() != rhs.getIndex(); }
    __host__ __device__ Message& operator++() { ++index;  return *this; }
    __host__ __device__ size_type getIndex() const { return this->index; };
    template<typename T, size_type N>
    __device__ T getVariable(const char(&variable_name)[N]);

    };

    // message list iterator inner class.
    class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {

    private:
        MessageList::Message _message;
    public:
        __host__ __device__ iterator(MessageList &messageList, size_type index) : _message(messageList, index) {}
        __host__ __device__ iterator& operator++() { ++_message;  return *this; }
        __host__ __device__ iterator operator++(int) { iterator tmp(*this); operator++(); return tmp; }
        __host__ __device__ bool operator==(const iterator& rhs) { return  _message == rhs._message; }
        __host__ __device__ bool operator!=(const iterator& rhs) { return  _message != rhs._message; }
        __host__ __device__ MessageList::Message& operator*() { return _message; }
    };

    __device__ MessageList(size_type start= 0, size_type end = 0) : start_(start), _messageCount(end) {};

    __device__ ~MessageList() {};

    /*! Returns the number of elements in the message list.
    */
    inline __host__ __device__ size_type size(void) const {

        return _messageCount;
    }

    inline __host__ __device__ iterator begin(void) { // const

        return iterator(*this, start_);
    }
    inline __host__ __device__ iterator end(void) { // const

        return iterator(*this, start_ + size());
    }

    template<typename T, size_type N> __device__
        T getVariable(MessageList::iterator iterator, const char(&variable_name)[N]);
    template<typename T, size_type N> __device__
        T getVariable(MessageList::Message message, const char(&variable_name)[N]);

    /**
    * \brief
    * \param agentname_hash
    */
    __device__ void setAgentNameSpace(CurveNamespaceHash agentname_hash) {

        agent_func_name_hash = agentname_hash;
    }

    /**
    * \brief
    * \param messagename_hash
    */
    __device__ void setMessageInpNameSpace(CurveNamespaceHash messagename_hash) {

        messagename_inp_hash = messagename_hash;
    }

    /**
    * \brief
    * \param  messageList_Size
    */
    __device__ void setMessageListSize(size_type messageCount) {

        _messageCount = messageCount;
    }

    __device__ CurveNamespaceHash getAgentNameSpace() {
        return this->agent_func_name_hash;
    }

    __device__ CurveNamespaceHash getMessageInpNameSpace() {
        return this->messagename_inp_hash;
    }

 private:

    size_type start_;
    size_type end_;

    size_type _messageCount;

    CurveNamespaceHash agent_func_name_hash;
    CurveNamespaceHash messagename_inp_hash;
};

/**
* \brief Gets an agent memory value given an iterator for the message list
* \param iterator MessageList iterator object for the target message.
* \param variable_name Name of memory variable to retrieve
*/
template<typename T, MessageList::size_type N>
__device__ T MessageList::getVariable(MessageList::iterator iterator, const char(&variable_name)[N]) {

    // get the value from curve using the message index.
    auto message = *iterator;
    T value = this->getVariable<T>(message, variable_name);
    return value;
}

/**
* \brief Gets an agent memory value given an iterator for the message list
* \param iterator MessageList iterator object for the target message.
* \param variable_name Name of memory variable to retrieve
*/
template<typename T, MessageList::size_type N>
__device__ T MessageList::getVariable(MessageList::Message message, const char(&variable_name)[N]) {

    // Ensure that the message is within bounds.
    if(message.getIndex() < this->_messageCount){

        // get the value from curve using the stored hashes and message index.
        T value = curveGetVariable<T>(variable_name, agent_func_name_hash + messagename_inp_hash, message.getIndex());
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return (T) 0;
    }
}

/**
* \brief Gets an agent memory value for a given MessageList::message
* \param variable_name Name of memory variable to retrieve
*/
template<typename T, MessageList::size_type N>
__device__ T MessageList::Message::getVariable(const char(&variable_name)[N]) {
    // Get the value from curve, using the internal messageList. 
    T value = _messageList.getVariable<T>(*this, variable_name);
    return value;
}


#endif /* MESSAGELIST_H_ */
