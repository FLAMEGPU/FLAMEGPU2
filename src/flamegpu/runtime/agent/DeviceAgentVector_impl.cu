#include <utility>
#include <vector>
#include <string>

#include "flamegpu/runtime/agent/DeviceAgentVector_impl.h"
#include "flamegpu/simulation/detail/CUDAAgent.h"
#include "flamegpu/runtime/agent/HostNewAgentAPI.h"

namespace flamegpu {

DeviceAgentVector_impl::DeviceAgentVector_impl(detail::CUDAAgent& _cuda_agent, const std::string &_cuda_agent_state,
                                               const VarOffsetStruct& _agentOffsets, std::vector<NewAgentStorage>& _newAgentData,
                                               detail::CUDAScatter& _scatter, const unsigned int _streamId, const cudaStream_t _stream)
    : AgentVector(_cuda_agent.getAgentDescription(), 0)
    , unbound_buffers_has_changed(false)
    , known_device_buffer_size(_cuda_agent.getStateSize(_cuda_agent_state))
    , cuda_agent(_cuda_agent)
    , cuda_agent_state(_cuda_agent_state)
    , agentOffsets(_agentOffsets)
    , newAgentData(_newAgentData)
    , scatter(_scatter)
    , streamId(_streamId)
    , stream(_stream) {
    // Create an empty AgentVector and initialise it manually
    // For each variable create an uninitialised array of variable data
    _size = known_device_buffer_size;
    internal_resize(_size, false);
    // Mark all variables as Invalid
    for (const auto& v : agent->variables)
        invalid_variables.insert(v.first);
    // Grab the unbound variable buffers from the CUDAFatAgentStateList
    // Leave their host counterparts de-allocated until required
    {
        const auto buffs = cuda_agent.getUnboundVariableBuffers(cuda_agent_state);
        for (auto &d_buff : buffs)
            unbound_buffers.emplace_back(d_buff);
        unbound_host_buffer_invalid = true;
    }
}

void DeviceAgentVector_impl::syncChanges() {
    // Resize device buffers if necessary
    const unsigned int old_allocated_size = cuda_agent.getStateAllocatedSize(cuda_agent_state);
    if (_size > old_allocated_size) {
        const unsigned int old_size = cuda_agent.getStateSize(cuda_agent_state);
        // Resize the underlying variable buffers for this agent state and retain variable data
        cuda_agent.resizeState(cuda_agent_state, _size, true, stream);  // @todo Don't retain data for mapped buffers?
        // Init agent data for any variables of newly created agents which are only present in a parent model
        const unsigned int new_allocated_size = cuda_agent.getStateAllocatedSize(cuda_agent_state);
        // This call does not use streams properly internally
        cuda_agent.initExcludedVars(cuda_agent_state, new_allocated_size - old_size, old_size, scatter, streamId, stream);
    }
    _requireLength();
    // Copy all changes back to device
    for (const auto &ch : change_detail) {
        auto &v = agent->variables.at(ch.first);
        // Copy back variable data into each array
        const char* host_src = static_cast<const char*>(_data->at(ch.first)->getDataPtr());
        char* device_dest = static_cast<char*>(cuda_agent.getStateVariablePtr(cuda_agent_state, ch.first));
        const size_t copy_offset = ch.second.first * v.type_size * v.elements;
        const size_t copy_len = (ch.second.second - ch.second.first) * v.type_size * v.elements;
        gpuErrchk(cudaMemcpyAsync(device_dest + copy_offset, host_src + copy_offset, copy_len, cudaMemcpyHostToDevice, stream));
    }
    change_detail.clear();
    // Copy all unbound buffes
    if (unbound_buffers_has_changed) {
        if (unbound_host_buffer_size != _size) {
            THROW exception::InvalidOperation("Unbound buffers have gone out of sync, in DeviceAgentVector::syncChanges().\n");
        }
        for (auto &buff : unbound_buffers) {
            const size_t variable_size = buff.device->type_size * buff.device->elements;
            gpuErrchk(cudaMemcpyAsync(buff.device->data, buff.host, unbound_host_buffer_size * variable_size, cudaMemcpyHostToDevice, stream));
        }
        unbound_buffers_has_changed = false;
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    // Update CUDAAgent statelist size
    cuda_agent.setStateAgentCount(cuda_agent_state, _size);
}
void DeviceAgentVector_impl::purgeCache() {
    _size = cuda_agent.getStateSize(cuda_agent_state);
    // All variables are now invalid
    for (const auto& v : agent->variables)
        invalid_variables.insert(v.first);
    // Mark all unbound host buffers as requiring update
    unbound_host_buffer_invalid = false;
    unbound_host_buffer_size = 0;
    known_device_buffer_size = cuda_agent.getStateSize(cuda_agent_state);
    unbound_buffers_has_changed = false;
}

void DeviceAgentVector_impl::initUnboundBuffers() {
    if (!_capacity)
      return;
    const unsigned int device_len = cuda_agent.getStateSize(cuda_agent_state);
    const unsigned int copy_len = _size < device_len ? _size : device_len;
    // Resize to match _capacity
    for (auto &buff : unbound_buffers) {
        if (buff.host) {
            THROW exception::InvalidOperation("Host buffer is already allocated, in DeviceAgentVector::initUnboundBuffers().\n");
        }
        // Alloc
        const size_t var_size = buff.device->type_size * buff.device->elements;
        buff.host = static_cast<char*>(malloc(_capacity * var_size));
        // DtH memcpy
        gpuErrchk(cudaMemcpyAsync(buff.host, buff.device->data, copy_len * var_size, cudaMemcpyDeviceToHost, stream));
        // Not sure this will ever happen, but better safe
        for (unsigned int i = device_len; i < _size; ++i) {
            // We have unknown agents, default init them
            memcpy(buff.host + i * var_size, buff.device->default_value, var_size);
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    unbound_host_buffer_capacity = _capacity;
    unbound_host_buffer_size = copy_len;
    unbound_buffers_has_changed = true;  // Probably not required, but if they are being init, high chance they're going to be changed
    unbound_host_buffer_invalid = false;
}
void DeviceAgentVector_impl::reinitUnboundBuffers() {
    const unsigned int device_len = cuda_agent.getStateSize(cuda_agent_state);
    const unsigned int copy_len = _size;
    if (device_len > _size) {
        THROW exception::InvalidOperation("Unexpected state, in DeviceAgentVector::reinitUnboundBuffers()\n");
    }
    // Resize to match _capacity
    for (auto& buff : unbound_buffers) {
        if (!buff.host) {
            THROW exception::InvalidOperation("Host buffer is not already allocated, in DeviceAgentVector::reinitUnboundBuffers().\n");
        }
        const size_t var_size = buff.device->type_size * buff.device->elements;
        if (unbound_host_buffer_capacity < _capacity) {
            free(buff.host);
            // Alloc
            buff.host = static_cast<char*>(malloc(_capacity * var_size));
        }
        // DtH memcpy
        gpuErrchk(cudaMemcpyAsync(buff.host, buff.device->data, copy_len * var_size, cudaMemcpyDeviceToHost, stream));
        // Not sure this will ever happen, but better safe
        for (unsigned int i = device_len; i < _size; ++i) {
            // We have unknown agents, default init them
            memcpy(buff.host + i * var_size, buff.device->default_value, var_size);
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    unbound_host_buffer_capacity = unbound_host_buffer_capacity < _capacity ?_capacity : unbound_host_buffer_capacity;
    unbound_host_buffer_size = copy_len;
    unbound_buffers_has_changed = true;  // Probably not required, but if they are being init, high chance they're going to be changed
    unbound_host_buffer_invalid = false;
}
void DeviceAgentVector_impl::resizeUnboundBuffers(const unsigned int new_capacity, bool init) {
    // Resize to match agent_count
    for (auto& buff : unbound_buffers) {
        if (!buff.host) {
            THROW exception::InvalidOperation("Not setup to resize before init");
        }
        // Alloc new buff
        const size_t var_size = buff.device->type_size * buff.device->elements;
        char *t = static_cast<char*>(malloc(new_capacity * var_size));
        // Copy data across
        const unsigned int copy_len = _size < unbound_host_buffer_capacity ? _size : unbound_host_buffer_capacity;
        memcpy(t, buff.host, copy_len * var_size);
        // Free old
        free(buff.host);
        // Replace old ptr
        buff.host = t;
        if (init) {
            for (unsigned int i = unbound_host_buffer_capacity; i < new_capacity; ++i) {
                // We have unknown agents, default init them
                memcpy(buff.host + i * var_size, buff.device->default_value, var_size);
            }
        }
    }
    unbound_host_buffer_capacity = new_capacity;
    // unbound_host_buffer_size = agent_count;  // This would only make sense for init, but consisent behaviour is better
    unbound_buffers_has_changed = true;  // Probably not required, but if they are resized, high chance theyre going to change
}

void DeviceAgentVector_impl::_insert(size_type pos, size_type count) {
    if (!count)
        return;
    // Init ID for all the inserted agents
    {
        auto d = _data->find(ID_VARIABLE_NAME);
        if (d != _data->end()) {
            _require(ID_VARIABLE_NAME);
            id_t *h_ptr = static_cast<id_t*>(d->second->getDataPtr());
            for (unsigned int i = pos; i < pos + count; ++i) {
                // Always assign ID, as AgentVector should reset these to unset, but this saves us checking
                // if (h_ptr[i] == ID_NOT_SET) {
                    h_ptr[i] = cuda_agent.nextID();
                // }
            }
            _changedAfter(ID_VARIABLE_NAME, pos);
        } else {
            THROW exception::InvalidOperation("Internal agent ID variable was not found, "
                "in DeviceAgentVector_impl._insert().");
        }
    }
    // No unbound buffers, return
    if (unbound_buffers.empty())
        return;
    // Unbound buffers first use, init
    // This updates unbound_host_buffer_size to match known_device_buffer_size
    if (!unbound_host_buffer_capacity)
        initUnboundBuffers();
    // Resizes unbound buffers if necessary
    const size_type new_size = known_device_buffer_size + count;
    if (new_size > unbound_host_buffer_capacity) {
        resizeUnboundBuffers(_capacity, false);
        // Init new agents that won't be init by the replacement below
        for (auto& buff : unbound_buffers) {
            const size_t variable_size = buff.device->type_size * buff.device->elements;
            for (unsigned int i = new_size; i < _capacity; ++i) {
                memcpy(buff.host + i * variable_size, buff.device->default_value, variable_size);
            }
        }
    }
    if (unbound_host_buffer_invalid) {
        // Redownload unbound buffers from device
        reinitUnboundBuffers();
    }
    //  Move all items behind pos, then init all the newly inserted
    for (auto& buff : unbound_buffers) {
        const size_t variable_size = buff.device->type_size * buff.device->elements;
        // Move all items after this index backwards count places
        for (unsigned int i = known_device_buffer_size - 1; i >= pos; --i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(buff.host + (i + count) * variable_size, buff.host + i * variable_size, variable_size);
        }
        // Default init the inserted variables
        for (unsigned int i = pos; i < pos + count; ++i) {
            memcpy(buff.host + i * variable_size, buff.device->default_value, variable_size);
        }
    }
    // Update size
    unbound_buffers_has_changed = true;
    unbound_host_buffer_size = new_size;
    known_device_buffer_size = _size;
    if (unbound_host_buffer_size != _size) {
        THROW exception::InvalidOperation("Unbound buffers have gone out of sync, in DeviceAgentVector::_insert().\n");
    }
    // Update change detail for all variables
    for (const auto& v : agent->variables) {
        // Does it exist in change map
        auto change = change_detail.find(v.first);
        if (change == change_detail.end()) {
            change_detail.emplace(v.first, std::pair<size_type, size_type>{pos, _size});
        } else {
            // Inclusive min bound
            change->second.first = change->second.first > pos ? pos : change->second.first;
            // Exclusive max bound
            change->second.second = _size;
        }
    }
}
void DeviceAgentVector_impl::_erase(size_type pos, size_type count) {
    // No unbound buffers, return
    if (unbound_buffers.empty() || !count)
        return;
    // Unbound buffers first use, init
    if (!unbound_host_buffer_capacity)
        initUnboundBuffers();
    if (unbound_host_buffer_invalid) {
        // Redownload unbound buffers from device
        reinitUnboundBuffers();
    }
    const size_type new_size = known_device_buffer_size - count;
    const size_type copy_start = pos + count;
    for (auto& buff : unbound_buffers) {
        const size_t variable_size = buff.device->type_size * buff.device->elements;
        // Move all items after this index forwards count places
        for (unsigned int i = copy_start; i < unbound_host_buffer_size; ++i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(buff.host + (i - count) * variable_size, buff.host + i * variable_size, variable_size);
        }
        // Default init the empty variables at the end
        for (unsigned int i = new_size; i < known_device_buffer_size; ++i) {
            memcpy(buff.host + i * variable_size, buff.device->default_value, variable_size);
        }
    }
    // Update size
    unbound_buffers_has_changed = true;
    unbound_host_buffer_size = new_size;
    known_device_buffer_size = _size;
    if (unbound_host_buffer_size != _size) {
        THROW exception::InvalidOperation("Unbound buffers have gone out of sync, in DeviceAgentVector::_erase().\n");
    }
    // Update change detail for all variables
    for (const auto &v : agent->variables) {
        // Does it exist in change map
        auto change = change_detail.find(v.first);
        if (change == change_detail.end()) {
            change_detail.emplace(v.first, std::pair<size_type, size_type>{pos, _size});
        } else {
            // Inclusive min bound
            change->second.first = change->second.first > pos ? pos : change->second.first;
            // Exclusive max bound
            change->second.second = _size;
        }
    }
}


void DeviceAgentVector_impl::_changed(const std::string& variable_name, size_type pos) {
    // Check the variable exists
    auto var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW exception::InvalidAgentVar("Variable %s was not found, "
            "in DeviceAgentVector::_changed()\n",
            variable_name.c_str());
    }
    // Does it exist in change map
    auto change = change_detail.find(variable_name);
    if (change == change_detail.end()) {
        change_detail.emplace(variable_name, std::pair<size_type, size_type>{pos, pos + 1});
    } else {
        // Inclusive min bound
        change->second.first = change->second.first > pos ? pos : change->second.first;
        // Exclusive max bound
        change->second.second = change->second.second <= pos ? pos + 1 : change->second.second;
    }
}
void DeviceAgentVector_impl::_changedAfter(const std::string& variable_name, size_type pos) {
    // Check the variable exists
    auto var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW exception::InvalidAgentVar("Variable %s was not found, "
            "in DeviceAgentVector::_changed()\n",
            variable_name.c_str());
    }
    // Does it exist in change map
    auto change = change_detail.find(variable_name);
    if (change == change_detail.end()) {
        change_detail.emplace(variable_name, std::pair<size_type, size_type>{pos, _size});
    } else {
        // Inclusive min bound
        change->second.first = change->second.first > pos ? pos : change->second.first;
        // Exclusive max bound
        change->second.second = _size;
    }
}
void DeviceAgentVector_impl::_require(const std::string& variable_name) const {
    if (invalid_variables.find(variable_name) !=invalid_variables.end()) {
        const auto& v = agent->variables.at(variable_name);
        // Copy back variable data into array
        void* host_dest = _data->at(variable_name)->getDataPtr();
        const void* device_src = cuda_agent.getStateVariablePtr(cuda_agent_state, variable_name);
        gpuErrchk(cudaMemcpyAsync(host_dest, device_src, _size * v.type_size * v.elements, cudaMemcpyDeviceToHost, stream));
        if (_capacity > _size) {
            // Default-init remaining buffer space
            const auto it = _data->find(variable_name);
            const size_t variable_size = v.type_size * v.elements;
            char* t_data = static_cast<char*>(it->second->getDataPtr());
            for (unsigned int i = _size; i < _capacity; ++i) {
                memcpy(t_data + i * variable_size, v.default_value, variable_size);
            }
        }
        // The invalid variable is now current
        invalid_variables.erase(variable_name);
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}
void DeviceAgentVector_impl::_requireAll() const {
    for (const auto& vn : invalid_variables) {
        const auto &v = agent->variables.at(vn);
        // Copy back variable data into array
        void* host_dest = _data->at(vn)->getDataPtr();
        const void* device_src = cuda_agent.getStateVariablePtr(cuda_agent_state, vn);
        gpuErrchk(cudaMemcpyAsync(host_dest, device_src, _size * v.type_size * v.elements, cudaMemcpyDeviceToHost, stream));
    }
    // Perform the cuda ops in a separate loop to host inits, gives a slight bit of time to eat latency
    for (const auto& vn : invalid_variables) {
        if (_capacity > _size) {
            const auto& v = agent->variables.at(vn);
            // Default-init remaining buffer space
            const auto it = _data->find(vn);
            const size_t variable_size = v.type_size * v.elements;
            char* t_data = static_cast<char*>(it->second->getDataPtr());
            for (unsigned int i = _size; i < _capacity; ++i) {
                memcpy(t_data + i * variable_size, v.default_value, variable_size);
            }
        }
    }
    // All invalid variables are now current
    invalid_variables.clear();
    gpuErrchk(cudaStreamSynchronize(stream));
}
void DeviceAgentVector_impl::_requireLength() const {
    /**
     * This method is a nightmare, as it needs to be const, so can't call non-const untility methods
     * Copy the implementations was bad, so I just decided to abuse const cast instead
     */
    if (newAgentData.empty())
        return;
    if (_size + newAgentData.size() > _capacity) {
        // BEGIN: Re implementation of AgentVector::resize(size_type, bool)
        // Can't call it here, as would have huge knock-on effects to which methods can/can't be const
        const_cast<DeviceAgentVector_impl*>(this)->internal_resize(_size + static_cast<size_type>(newAgentData.size()), false);
        // END: Re implementation of AgentVector::resize(size_type, bool)
    }
    _requireAll();
    // Check if host new agent has any agents
    for (auto &newAgent : newAgentData) {
        // Manually insert them to device agent vector
        for (auto &v : agentOffsets.vars) {
            char* dst = static_cast<char*>(_data->at(v.first)->getDataPtr()) + _size * v.second.len;
            const char * src = newAgent.data + v.second.offset;
            memcpy(dst, src, v.second.len);
        }
        // Increase size
        ++_size;
    }
    // This updates unbound buffers
    // BEGIN: Re implementation of DeviceAgentVector_t::_insert(size_type, size_type)
    // Can't call it here, as would have huge knock-on effects to which methods can/can't be const
    const_cast<DeviceAgentVector_impl*>(this)->_insert(_size - static_cast<size_type>(newAgentData.size()), static_cast<size_type>(newAgentData.size()));
    // END: Re implementation of DeviceAgentVector_t::_insert(size_type, size_type)
    newAgentData.clear();
}


}  // namespace flamegpu
