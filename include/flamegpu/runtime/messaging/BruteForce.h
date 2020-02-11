#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_

#include "flamegpu/runtime/cuRVE/curve.h"

class MsgBruteForce
{
    friend class MessageList;
    friend class Message;
    typedef unsigned int size_type;
public:
    class Message;  // Forward declare inner classes
    class iterator;  // Forward declare inner classes
                     //Unused, just required for some lazy template initialising
    MsgBruteForce() {}
    __device__ MsgBruteForce(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int _len)
        : combined_hash(agentfn_hash + msg_hash)
        , len(_len)
    {

    }
    /**
    * This class is returned to user by Device API
    * It gives access to message iterators
    */
    class MessageList
    {
    public:
        MessageList(MsgBruteForce &_parent)
            : parent(_parent)
        { }
        // Something to access messages (probably iterator rather than get var
        /*! Returns the number of elements in the message list.
        */
        inline __host__ __device__ size_type size(void) const {
            return parent.len;
        }

        inline __host__ __device__ iterator begin(void) {  // const
            return iterator(parent, 0);
        }
        inline __host__ __device__ iterator end(void) {  // const
                                                         //If there can be many begin, each with diff end, we need a middle layer to host the iterator/s
            return iterator(parent, parent.len);
        }
    private:
        MsgBruteForce &parent;
    };
    // Inner class representing an individual message
    class Message {
    private:
        MsgBruteForce &_parent;
        size_type index;
    public:
        __device__ Message(MsgBruteForce &parent) : _parent(parent), index(0) {}
        __device__ Message(MsgBruteForce &parent, size_type index) : _parent(parent), index(index) {}
        __host__ __device__ bool operator==(const Message& rhs) { return  this->getIndex() == rhs.getIndex(); }
        __host__ __device__ bool operator!=(const Message& rhs) { return  this->getIndex() != rhs.getIndex(); }
        __host__ __device__ Message& operator++() { ++index;  return *this; }
        __host__ __device__ size_type getIndex() const { return this->index; }
        template<typename T, size_type N>
        __device__ T getVariable(const char(&variable_name)[N]);
    };
    // message list iterator inner class.
    class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
    private:
        MsgBruteForce::Message _message;
    public:
        __host__ __device__ iterator(MsgBruteForce &parent, size_type index) : _message(parent, index) {}
        __host__ __device__ iterator& operator++() { ++_message;  return *this; }
        __host__ __device__ iterator operator++(int) { iterator tmp(*this); operator++(); return tmp; }
        __host__ __device__ bool operator==(const iterator& rhs) { return  _message == rhs._message; }
        __host__ __device__ bool operator!=(const iterator& rhs) { return  _message != rhs._message; }
        __host__ __device__ MsgBruteForce::Message& operator*() { return _message; }
    };
private:
    //agent_function + msg_hash
    Curve::NamespaceHash combined_hash;
    size_type len;
};
template<typename T, unsigned int N>
__device__ T MsgBruteForce::Message::getVariable(const char(&variable_name)[N]) {
    // Ensure that the message is within bounds.
    if (index < this->_parent.len) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, index);
        return value;
    }
    else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_