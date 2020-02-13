#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_

#include "flamegpu/runtime/cuRVE/curve.h"

class MsgBruteForce {
    typedef unsigned int size_type;

 public:
    class Message;   // Forward declare inner classes
    class iterator;  // Forward declare inner classes
    struct MetaData {
        unsigned int length;
    };
    /**
    * This class is returned to user by Device API
    * It gives access to message iterators
    */
    class In {
        friend class MsgBruteForce::Message;

     public:
        __device__ In(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, const void *metadata)
            : combined_hash(agentfn_hash + msg_hash)
            , len(reinterpret_cast<const MetaData*>(metadata)->length)
        { }
        // Something to access messages (probably iterator rather than get var
        /*! Returns the number of elements in the message list.
        */
        inline __host__ __device__ size_type size(void) const {
            return len;
        }

        inline __host__ __device__ iterator begin(void) const {  // const
            return iterator(*this, 0);
        }
        inline __host__ __device__ iterator end(void) const  {  // const
            // If there can be many begin, each with diff end, we need a middle layer to host the iterator/s
            return iterator(*this, len);
        }

     private:
        // agent_function + msg_hash
        Curve::NamespaceHash combined_hash;
        size_type len;
    };
    // Inner class representing an individual message
    class Message {
     private:
        const MsgBruteForce::In &_parent;
        size_type index;

     public:
        __device__ Message(const MsgBruteForce::In &parent) : _parent(parent), index(0) {}
        __device__ Message(const MsgBruteForce::In &parent, size_type index) : _parent(parent), index(index) {}
        __host__ __device__ bool operator==(const Message& rhs) { return  this->getIndex() == rhs.getIndex(); }
        __host__ __device__ bool operator!=(const Message& rhs) { return  this->getIndex() != rhs.getIndex(); }
        __host__ __device__ Message& operator++() { ++index;  return *this; }
        __host__ __device__ size_type getIndex() const { return this->index; }
        template<typename T, size_type N>
        __device__ T getVariable(const char(&variable_name)[N]) const;
    };
    // message list iterator inner class.
    class iterator : public std::iterator <std::random_access_iterator_tag, void, void, void, void> {
     private:
        MsgBruteForce::Message _message;

     public:
        __host__ __device__ iterator(const MsgBruteForce::In &parent, size_type index) : _message(parent, index) {}
        __host__ __device__ iterator& operator++() { ++_message;  return *this; }
        __host__ __device__ iterator operator++(int) { iterator tmp(*this); operator++(); return tmp; }
        __host__ __device__ bool operator==(const iterator& rhs) { return  _message == rhs._message; }
        __host__ __device__ bool operator!=(const iterator& rhs) { return  _message != rhs._message; }
        __host__ __device__ MsgBruteForce::Message& operator*() { return _message; }
    };

    class Out {
     public:
        __device__ Out(Curve::NamespaceHash agentfn_hash, Curve::NamespaceHash msg_hash, unsigned int _streamId)
            : combined_hash(agentfn_hash + msg_hash)
            , streamId(_streamId)
        { }
        template<typename T, unsigned int N>
        __device__ void setVariable(const char(&variable_name)[N], T value) const;

     private:
        // agent_function + msg_hash
        Curve::NamespaceHash combined_hash;
        unsigned int streamId;
    };

    // Blank handler, brute force requires no index or special allocations
    template<typename SimSpecialisationMsg>
    class CUDAModelHandler : public MsgSpecialisationHandler<SimSpecialisationMsg> {
     public:
        explicit CUDAModelHandler(CUDAMessage &a)
            : MsgSpecialisationHandler(a) {
            gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
        }
        ~CUDAModelHandler() {
            gpuErrchk(cudaFree(d_metadata));
        }

        void buildIndex() override {
            hd_metadata.length = sim_message.getMessageCount();
            gpuErrchk(cudaMemcpy(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice));
        }
        const void *getMetaDataDevicePtr() const override { return d_metadata; }

     private:
        MetaData hd_metadata;
        MetaData *d_metadata;
    };
};
template<typename T, unsigned int N>
__device__ T MsgBruteForce::Message::getVariable(const char(&variable_name)[N]) const {
    // Ensure that the message is within bounds.
    if (index < this->_parent.len) {
        // get the value from curve using the stored hashes and message index.
        T value = Curve::getVariable<T>(variable_name, this->_parent.combined_hash, index);
        return value;
    } else {
        // @todo - Improved error handling of out of bounds message access? Return a default value or assert?
        return static_cast<T>(0);
    }
}

/**
* \brief adds a message
* \param variable_name Name of message variable to set
* \param value Value to set it to
*/
template<typename T, unsigned int N>
__device__ void MsgBruteForce::Out::setVariable(const char(&variable_name)[N], T value) const {  // message name or variable name
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // Todo: checking if the output message type is single or optional?  (d_message_type)

    // set the variable using curve
    Curve::setVariable<T>(variable_name, combined_hash, value, index);

    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].scan_flag[index] = 1;
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_BRUTEFORCE_H_
