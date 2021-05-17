#ifndef INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_IMPL_H_
#define INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_IMPL_H_

#include <string>
#include <utility>
#include <memory>
#include <list>
#include <map>
#include <set>
#include <vector>

#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/gpu/CUDAFatAgentStateList.h"  // VariableBuffer

class CUDAScatter;
class CUDAAgent;

struct VarOffsetStruct;
struct NewAgentStorage;

/**
 * This class provides an AgentVector interface to agent data currently stored on the device during execution of a CUDASimulation
 *
 * It attempts to prevent unnecessary memory transfers, as copying all agent variable buffers to only use 1
 * Would result in a large number of redundant but costly memcpys between host and device
 *
 * @note This class cannot be created directly, you should use one created by HostAgentAPI
 * @see HostAgentAPI::getPopulationData()
 */
class DeviceAgentVector_impl : protected AgentVector {
 public:
    /**
     * Construct a DeviceAgentVector interface to the on-device data of cuda_agent
     * @param _cuda_agent CUDAAgent instance holding pointers to the desired agent data
     * @param cuda_agent_state Name of the state within cuda_agent to represent.
     * @param _agentOffsets Agent offset metadata for storing variable data into NewAgentStorage
     * @param _newAgentData Vector of NewAgentStorage to automatically perform host agent birth when appropriate to save later host-device memcpys
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    DeviceAgentVector_impl(CUDAAgent& _cuda_agent, const std::string& cuda_agent_state,
        const VarOffsetStruct& _agentOffsets, std::vector<NewAgentStorage>& _newAgentData,
        CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream);
    /**
     * Copy operations are disabled
     */
     // DeviceAgentVector_t(const DeviceAgentVector_t& other) = delete;
     // DeviceAgentVector_t& operator=(const DeviceAgentVector_t& other) = delete;
     /**
      * Copies changed agent data back to device
      */
    void syncChanges();
    /**
     * Clears the local cache, so data is re-downloaded from the device when next required
     */
    void purgeCache();

    /**
     * Access specified element with bounds checking
     * @param pos position of the element to return
     * @return Reference to the requested element.
     * @throws std::out_of_range if `!(pos < size())`
     */
    using AgentVector::at;
    /**
     * Returns a reference to the element at specified location pos.
     * @param pos position of the element to return
     * @return Reference to the requested element.
     * @throws std::out_of_range if `!(pos < size())`
     */
    using AgentVector::operator[];
    /**
     * Returns a reference to the first element in the container.
     * @return Reference to the first element
     * @throws std::out_of_range if `empty()`
     */
    using AgentVector::front;
    /**
     * Returns a reference to the last element in the container.
     * @return Reference to the last element
     * @throws std::out_of_range if `empty()`
     */
    using AgentVector::back;
    // using AgentVector::data; // Would need to assume whole vector changed
    /**
     * Forward iterator access to the start of the vector
     */
    using AgentVector::begin;
    /**
     * Forward iterator to position after the last element
     */
    using AgentVector::end;
    /**
     * Forward iterator const access to the start of the vector
     */
    using AgentVector::cbegin;
    /**
     * Forward iterator to position after the last element
     * (Const version of end())
     */
    using AgentVector::cend;
    /**
     * Reverse iterator access to the last element of the vector
     */
    using AgentVector::rbegin;
    /**
     * Reverse iterator to position before the first element
     */
    using AgentVector::rend;
    /**
     * Reverse iterator const access to the last element of the vector
     */
    using AgentVector::crbegin;
    /**
     * Reverse iterator to position before the first element
     * (Const version of rend())
     */
    using AgentVector::crend;
    /**
     * Checks if the container has no elements, i.e. whether begin() == end()
     * @return `true` if the container is empty, `false` otherwise
     */
    using AgentVector::empty;
    /**
     * Returns the number of elements in the container, i.e. std::distance(begin(), end())
     * @return The number of elements in the container.
     */
    using AgentVector::size;
    /**
     * Returns the max theoretical size of this host vector
     * This does not reflect the device allocated buffer
     */
    using AgentVector::max_size;
    /**
     * Pre-allocates buffer space for this host vector
     * This does not affect the device allocated buffer, that is updated if necessary when agents are returned to device.
     */
    using AgentVector::reserve;
    /**
     * Returns the current capacity of the host vector
     * This does not reflect the capacity of the device allocated buffer
     */
    using AgentVector::capacity;
    /**
     * Reduces the current capacity to fit the size of the host vector
     * This does not affect the capacity of the device allocated buffer, that is updated if necessary when agents are returned to device.
     */
    using AgentVector::shrink_to_fit;
    /**
     * Erases all elements from the container. After this call, size() returns zero.
     *
     * Invalidates any references, pointers, or iterators referring to contained elements. Any past-the-end iterators are also invalidated.
     *
     * Leaves the capacity() of the vector unchanged
     */
    using AgentVector::clear;
    /**
     * Inserts elements at the specified location in the container
     * a) Inserts value before pos
     * b) Inserts count copies of the value before pos
     * c) Inserts elements from range [first, last) before pos.
     *    The behavior is undefined if first and last are iterators into *this
     *
     * Causes reallocation if the new size() is greater than the old capacity().
     * If the new size() is greater than capacity(), all iterators and references are invalidated.
     * Otherwise, only the iterators and references before the insertion point remain valid.
     * The past-the-end iterator is also invalidated.
     *
     * @throw InvalidAgent If agent type of value does not match
     * @note Inserted agents will be assigned a new unique ID
     */
    using AgentVector::insert;
#ifdef SWIG
    /**
     * Python insert behaviour
     */
    using AgentVector::py_insert;
#endif
    /**
     * Erases the specified elements from the container.
     * a) Removes the element at pos.
     * b) Removes the elements in the range [first, last).
     *
     * Invalidates iterators and references at or after the point of the erase,  including the end() iterator.
     *
     * The iterator pos must be valid and dereferenceable.
     * Thus the end() iterator (which is valid, but is not dereferenceable) cannot be used as a value for pos.
     *
     * @param pos iterator to the element to remove
     *
     * @return Iterator following the last removed element
     * @return If pos refers to the last element, then the end() iterator is returned
     * @throw OutOfBoundsException pos >= size()
     */
    using AgentVector::erase;
#ifdef SWIG
    /**
     * Python erase behaviour
     */
    using AgentVector::py_erase;
#endif
    /**
     * Appends a default initialised agent to the end of the container
     * @note Inserted agent will be assigned a new unique ID
     */
    using AgentVector::push_back;
    /**
     * Removes the last agent of the container.
     * Calling pop_back on an empty container results in undefined behavior.
     * Iterators and references to the last element, as well as the end() iterator, are invalidated.
     */
    using AgentVector::pop_back;
    using AgentVector::resize;
    // using AgentVector::swap; // This would essentially require replacing the entire on-device agent vector

 protected:
    /**
     * Triggered when insert() has been called
     */
    void _insert(size_type pos, size_type count) override;
    /**
     * Triggered when erase() has been called
     */
    void _erase(size_type pos, size_type count) override;
    /**
     * Useful for notifying changes due when a single agent variable has been updated (AgentVector::Agent::setVariable())
     * @param variable_name Name of the variable that has been changed
     * @param pos The index of the agent that's variable has been changed
     */
    void _changed(const std::string& variable_name, size_type pos) override;
    /**
     * Useful for notifying changes due to inserting/removing items, which essentially move all trailing items
     * @param variable_name Name of the variable that has been changed
     * @param pos The first index that has been changed
     */
    void _changedAfter(const std::string& variable_name, size_type pos) override;
    /**
     * Notify this that a variable is about to be accessed, to allow it's data to be synced
     * Should be called by operations which update variables (e.g. AgentVector::Agent::getVariable())
     * @param variable_name Name of the variable that has been changed
     */
    void _require(const std::string& variable_name) const override;
    /**
     * Notify this that all variables are about to be accessed
     * Should be called by operations which move agents (e.g. insert/erase)
     * @note This is not called in conjunction with _insert() or _erase()
     */
    void _requireAll() const override;
    /**
     * Notify that the size is about to be accessed
     * This reflects both whether size is accessed directly or indirectly
     * This will poll HostNewAgent for creations and apply them to the data structure
     */
    void _requireLength() const override;
    /**
     * Store information regarding which variables have been changed
     * This map is built as changes come in, it is empty if no changes have been made
     */
    std::map<std::string, std::pair<size_type, size_type>> change_detail;
    /**
     * Variables included here require data to be updated from the device
     * @note Mutable, because it must be updated by _requires(), _requiresAll() which are const
     *       as they can be called by const user methods
     */
    mutable std::set<std::string> invalid_variables;
    /**
     * Store information regarding which variables have been changed
     * This map is built as changes come in, it is empty if no changes have been made
     */
    bool unbound_buffers_has_changed;

 private:
    /**
     * Pair of a host-backed device buffer
     * This allows transactions which impact master-agent unbound variables to work correctly
     * @note Can't move this definition to .cu, gcc requires it for std::list
     */
     struct VariableBufferPair {
         /**
          * Never allocate
         */
         explicit VariableBufferPair(const std::shared_ptr<VariableBuffer>& _device)
             : device(_device) { }
         VariableBufferPair(VariableBufferPair&& other) {
             *this = std::move(other);
         }
         VariableBufferPair& operator=(VariableBufferPair&& other) {
             std::swap(this->host, other.host);
             std::swap(this->device, other.device);
             return *this;
         }
         /**
         * Copy operations are disabled
         */
         // @todo Solve this
         // VariableBufferPair(const VariableBufferPair& other) = delete;
         // VariableBufferPair& operator=(const VariableBufferPair& other) = delete;

         /**
          * Free host if
          */
         ~VariableBufferPair() {
             if (host) free(host);
         }
         /**
         * nullptr until required to be allocated
         */
         char* host = nullptr;
         /**/
         std::shared_ptr<VariableBuffer> device;
     };
    /**
     * Any operations which move agents just be applied to this buffers too
     */
    std::list<VariableBufferPair> unbound_buffers;
    /**
     * The currently known size of the device buffer
     * This is used to track size before the unbound_buffers are init
     * Can't use _size in place of this, as calls to insert/erase/etc update that before we are notified
     */
    unsigned int known_device_buffer_size = 0;
    /**
     * Number of agents currently allocated inside the host_buffers
     * Useful if the device buffers are resized via the CUDAAgent
     */
    unsigned int unbound_host_buffer_size = 0;
    unsigned int unbound_host_buffer_capacity = 0;
    /**
     * This is set true by clearCache()
     * If data has been re-ordered on the device, the host buffers will be out of sync
     * At next insert/erase, this tells host buffers to download new
     * It also tells a future call to sync, to ignore the unbound host buffers
     */
    bool unbound_host_buffer_invalid = false;
    /**
     * Initialises the host copies of the unbound buffers
     * Allocates the host copy, and copies device data to them
     */
    void initUnboundBuffers();
    /**
     * Re-downloads updates the host unbound buffers from the device
     */
    void reinitUnboundBuffers();
    /**
     * Resizes the host copy of the unbound buffers, retaining data
     * @param new_capacity New buffer capacity
     * @param init If true, new memory is init
     */
    void resizeUnboundBuffers(const unsigned int& new_capacity, bool init);
    CUDAAgent& cuda_agent;
    std::string cuda_agent_state;


    const VarOffsetStruct& agentOffsets;
    std::vector<NewAgentStorage>& newAgentData;

    CUDAScatter& scatter;
    const unsigned int& streamId;
    const cudaStream_t& stream;
};

#endif  // INCLUDE_FLAMEGPU_POP_DEVICEAGENTVECTOR_IMPL_H_
