#include "flamegpu/pop/AgentVector.h"

#include "flamegpu/model/AgentDescription.h"

#include "flamegpu/pop/AgentVector_Agent.h"

#ifdef max
#undef max  // Unclear where this definition is leaking from
#endif

const float AgentVector::RESIZE_FACTOR = 1.5f;

AgentVector::AgentVector(const AgentDescription& agent_desc, size_type count)
    : agent(agent_desc.agent->clone())
    , _size(0)
    , _capacity(0)
    , _data(std::make_shared<AgentDataMap>()) {
    resize(count);
}

AgentVector::AgentVector(const AgentVector& other)
    : agent(other.agent->clone())
    , _size(0)
    , _capacity(0)
    , _data(std::make_shared<AgentDataMap>()) {
    clear();
    *this = other;
}

AgentVector::AgentVector(AgentVector&& other) noexcept
    : agent(other.agent->clone())
    , _size(other._size)
    , _capacity(other._capacity)
    , _data(std::make_shared<AgentDataMap>()) {
    // Purge our data
    _data->clear();
    // Swap data
    std::swap(*_data, *other._data);  // Not 100% sure this will work as intended
    // Purge other
    other._size = 0;
    other._capacity = 0;
}

AgentVector& AgentVector::operator=(const AgentVector& other) {
    // Self assignment
    if (this == &other)
        return *this;
    if (*agent != *other.agent) {
        throw std::exception();  // AgentVectors are for different AgentDescriptions
    }
    // Copy size
    resize(other.size(), false);
    _size = other.size();
    if (_size) {
        // Copy data
        for (const auto& v : agent->variables) {
            auto &this_it = _data->at(v.first);
            const auto &other_it = other._data->at(v.first);
            memcpy(this_it->getDataPtr(), other_it->getReadOnlyDataPtr(), _size * v.second.type_size * v.second.elements);
        }
    }
    return *this;
}
AgentVector& AgentVector::operator=(AgentVector&& other) noexcept {
    agent = other.agent->clone();
    _size = other._size;
    _capacity = other._capacity;
    // Purge our data
    _data->clear();
    // Swap data
    std::swap(*_data, *other._data);  // Not 100% sure this will work as intended
    // Purge other
    other._size = 0;
    other._capacity = 0;
    return *this;
}

AgentVector::Agent AgentVector::at(size_type pos) {
    if (pos >= size()) {
        THROW OutOfBoundsException("pos (%u) exceeds length of vector (%u) in AgentVector::at()", pos, size());
    }
    // Return the agent instance
    return Agent(agent, _data, pos);
}
AgentVector::CAgent AgentVector::at(size_type pos) const {
    if (pos >= size()) {
        THROW OutOfBoundsException("pos (%u) exceeds length of vector (%u) in AgentVector::at()", pos, size());
    }
    return CAgent(agent, _data, pos);
}

AgentVector::Agent AgentVector::operator[](size_type pos) {
    return at(pos);
}
AgentVector::CAgent AgentVector::operator[](size_type pos) const {
    return at(pos);
}

AgentVector::Agent AgentVector::front() {
    return at(0);
}
AgentVector::CAgent AgentVector::front() const {
    return at(0);
}
AgentVector::Agent AgentVector::back() {
    return at(_size - 1);
}
AgentVector::CAgent AgentVector::back() const {
    return at(_size - 1);
}

void* AgentVector::data(const std::string& variable_name) {
    // Is variable name found
    const auto& var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent '%s', "
            "in AgentVector::data().",
            variable_name.c_str(), agent->name.c_str());
    }
    // Does the map have a vector
    const auto& map_it = _data->find(variable_name);
    if (map_it != _data->end())
        return map_it->second->getDataPtr();
    return nullptr;
}
const void* AgentVector::data(const std::string& variable_name) const {
    // Is variable name found
    const auto& var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent '%s', "
            "in AgentVector::data().",
            variable_name.c_str(), agent->name.c_str());
    }
    // Does the map have a vector
    const auto& map_it = _data->find(variable_name);
    if (map_it != _data->end())
        return map_it->second->getDataPtr();
    return nullptr;
}

AgentVector::iterator AgentVector::begin() noexcept {
    return iterator(agent, _data, 0);
}
AgentVector::const_iterator AgentVector::begin() const noexcept {
    return const_iterator(agent, _data, 0);
}
AgentVector::const_iterator AgentVector::cbegin() const noexcept {
    return const_iterator(agent, _data, 0);
}
AgentVector::iterator AgentVector::end() noexcept {
    return iterator(agent, _data, _size);
}
AgentVector::const_iterator AgentVector::end() const noexcept {
    return const_iterator(agent, _data, _size);
}
AgentVector::const_iterator AgentVector::cend() const noexcept {
    return const_iterator(agent, _data, _size);
}
AgentVector::reverse_iterator AgentVector::rbegin() noexcept {
    return reverse_iterator(agent, _data, _size - 1);
}
AgentVector::const_reverse_iterator AgentVector::rbegin() const noexcept {
    return const_reverse_iterator(agent, _data, _size-1);
}
AgentVector::const_reverse_iterator AgentVector::crbegin() const noexcept {
    return const_reverse_iterator(agent, _data, _size-1);
}
AgentVector::reverse_iterator AgentVector::rend() noexcept {
    return reverse_iterator(agent, _data, std::numeric_limits<size_type>::max());
}
AgentVector::const_reverse_iterator AgentVector::rend() const noexcept {
    return const_reverse_iterator(agent, _data, std::numeric_limits<size_type>::max());
}
AgentVector::const_reverse_iterator AgentVector::crend() const noexcept {
    return const_reverse_iterator(agent, _data, std::numeric_limits<size_type>::max());
}

bool AgentVector::empty() const { return _size == 0; }
AgentVector::size_type AgentVector::size() const { return _size; }
AgentVector::size_type AgentVector::max_size() { return std::numeric_limits<size_type>::max() - 1; }
void AgentVector::reserve(size_type new_cap) {
    if (new_cap > _capacity) {
        resize(new_cap, true);
    }
}
AgentVector::size_type AgentVector::capacity() const { return _capacity; }
void AgentVector::shrink_to_fit() {
    if (_size > _capacity) {
        resize(_size, false);
    }
}
void AgentVector::clear() {
    // Re initialise all variables
    if (_capacity) {
        for (const auto& v : agent->variables) {
            const auto it = _data->find(v.first);
            const size_t variable_size = v.second.type_size * v.second.elements;
            char* t_data = static_cast<char*>(it->second->getDataPtr());
            for (unsigned int i = 0; i < _size; ++i) {
                memcpy(t_data + i * variable_size, v.second.default_value, variable_size);
            }
        }
    }
    _size = 0;
}

AgentVector::iterator AgentVector::insert(const_iterator pos, const AgentInstance& value) {
    return insert(pos, 1, value);
}
AgentVector::iterator AgentVector::insert(size_type pos, const AgentInstance& value) {
    return insert(pos, 1, value);
}
AgentVector::iterator AgentVector::insert(const_iterator pos, const Agent& value) {
    return insert(pos, 1, value);
}
AgentVector::iterator AgentVector::insert(size_type pos, const Agent& value) {
    return insert(pos, 1, value);
}
AgentVector::iterator AgentVector::insert(const_iterator pos, size_type count, const AgentInstance& value) {
    return insert(pos._pos, count, value);
}
AgentVector::iterator AgentVector::insert(size_type pos, size_type count, const AgentInstance& value) {
    if (count == 0)
        return iterator(agent, _data, pos);
    // Confirm they are for the same agent type
    if (value._agent != agent && *value._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
            "in AgentVector::push_back().\n",
            value._agent->name.c_str(), agent->name.c_str());
    }
    // Expand capacity if required
    {
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + count > new_capacity) {
            new_capacity = static_cast<size_type>(new_capacity * RESIZE_FACTOR) + 1;
        }
        resize(new_capacity, true);
    }
    // Get first index;
    const size_type insert_index = pos;
    // Fix each variable
    for (const auto& v : agent->variables) {
        const auto it = _data->find(v.first);
        char* t_data = static_cast<char*>(it->second->getDataPtr());
        const size_t variable_size = v.second.type_size * v.second.elements;
        // Move all items after this index backwards count places
        for (unsigned int i = _size - 1; i >= insert_index; --i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(t_data + (i + count) * variable_size, t_data + i * variable_size, variable_size);
        }
        // Copy across item data
        const auto other_it = value._data.find(v.first);
        for (unsigned int i = insert_index; i < insert_index + count; ++i) {
            memcpy(t_data + i * variable_size, other_it->second.ptr, variable_size);
        }
    }
    // Increase size
    _size += count;
    // Return iterator to first inserted item
    return iterator(agent, _data, insert_index);
}
AgentVector::iterator AgentVector::insert(const_iterator pos, size_type count, const Agent& value) {
    return insert(pos._pos, count, value);
}
AgentVector::iterator AgentVector::insert(size_type pos, size_type count, const Agent& value) {
    if (count == 0)
        return iterator(agent, _data, pos);
    // Confirm they are for the same agent type
    if (value._agent != agent && *value._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
            "in AgentVector::push_back().\n",
            value._agent->name.c_str(), agent->name.c_str());
    }
    // Expand capacity if required
    {
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + count > new_capacity) {
            new_capacity = static_cast<size_type>(new_capacity * RESIZE_FACTOR) + 1;
        }
        resize(new_capacity, true);
    }
    // Get first index;
    const size_type insert_index = pos;
    // Fix each variable
    auto value_data = value._data.lock();
    if (!value_data) {
        THROW ExpiredWeakPtr("The AgentVector which owns the passed AgentVector::Agent has been deallocated, "
            "in AgentVector::insert().\n");
    }
    for (const auto& v : agent->variables) {
        const auto it = _data->find(v.first);
        char* t_data = static_cast<char*>(it->second->getDataPtr());
        const size_t variable_size = v.second.type_size * v.second.elements;
        // Move all items after this index backwards count places
        for (unsigned int i = _size - 1; i >= insert_index; --i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(t_data + (i + count) * variable_size, t_data + i * variable_size, variable_size);
        }
        // Copy across item data
        const auto other_it = value_data->find(v.first);
        char* src_data = static_cast<char*>(other_it->second->getDataPtr());
        for (unsigned int i = insert_index; i < insert_index + count; ++i) {
            memcpy(t_data + i * variable_size, src_data + value.index * variable_size, variable_size);
        }
    }
    // Increase size
    _size += count;
    // Return iterator to first inserted item
    return iterator(agent, _data, insert_index);
}
AgentVector::iterator AgentVector::erase(const_iterator pos) {
    const auto first = pos++;
    return erase(first._pos, pos._pos);
}
AgentVector::iterator AgentVector::erase(size_type pos) {
    const auto first = pos++;
    return erase(first, pos);
}
AgentVector::iterator AgentVector::erase(const_iterator first, const_iterator last) {
    // Confirm they are for the same agent type
    if (first._agent != agent && *first._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
            "in AgentVector::push_back().\n",
            first._agent->name.c_str(), agent->name.c_str());
    }
    if (last._agent != agent && *last._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
            "in AgentVector::push_back().\n",
            last._agent->name.c_str(), agent->name.c_str());
    }
    return erase(first._pos, last._pos);
}
AgentVector::iterator AgentVector::erase(size_type first, size_type last) {
    if (first == last)
        return iterator(agent, _data, last);
    // Get first index;
    const size_type first_remove_index = first< last ? first : last;
    const size_type first_move_index = first < last ? last : first;
    const size_type erase_count = first_move_index - first_remove_index;
    const size_type first_empty_index = _size - erase_count;
    const size_type last_empty_index = _size;
    // Ensure indicies are in bounds
    if (first_remove_index >= _size) {
        THROW OutOfBoundsException("%u is not a valid index into the vector, "
          "in AgentVector::erase()\n", first_remove_index);
    } else if (first_move_index > _size) {
        THROW OutOfBoundsException("%u is not a valid index to the end of a range of vector items, "
            "it must point to after the final selected item, "
            "in AgentVector::erase()\n", first_move_index);
    }
    // Fix each variable
    for (const auto& v : agent->variables) {
        const auto it = _data->find(v.first);
        char* t_data = static_cast<char*>(it->second->getDataPtr());
        const size_t variable_size = v.second.type_size * v.second.elements;
        // Move all items after this index forwards count places
        for (unsigned int i = first_move_index; i < _size; ++i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(t_data + (i - erase_count) * variable_size, t_data + i * variable_size, variable_size);
        }
        // Initialise newly empty variables
        for (unsigned int i = first_empty_index; i < last_empty_index; ++i) {
            memcpy(t_data + i * variable_size, v.second.default_value, variable_size);
        }
    }
    // Decrease size
    _size -= erase_count;
    // Return iterator following the last removed element
    return iterator(agent, _data, first_remove_index + 1);
}
void AgentVector::push_back(const AgentInstance& value) {
    insert(cend(), value);
}
void AgentVector::push_back() {
    // Expand capacity if required
    {
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + 1 > new_capacity) {
            new_capacity = static_cast<size_type>(_capacity * RESIZE_FACTOR) + 1;
        }
        resize(new_capacity, true);
    }
    ++_size;
}
void AgentVector::pop_back() {
    if (_size) {
        --_size;
        // Reset removed item to default value
        for (const auto& v : agent->variables) {
            const auto it = _data->find(v.first);
            const size_t variable_size = v.second.type_size * v.second.elements;
            char* t_data = static_cast<char*>(it->second->getDataPtr());
            memcpy(t_data + _size * variable_size, v.second.default_value, variable_size);
        }
    }
}
void AgentVector::resize(size_type count) {
    resize(count, true);
    _size = count;
}
void AgentVector::resize(size_type count, bool init) {
    if (count == _capacity)
        return;
    for (const auto& v : agent->variables) {
        // For each variable inside agent, add it to the map or replace it in the map
        const auto it = _data->find(v.first);
        const size_t variable_size = v.second.type_size * v.second.elements;
        if (it == _data->end()) {
            // Need to create the variable's vector
            auto t = std::unique_ptr<GenericMemoryVector>(v.second.memory_vector->clone());
            t->resize(count);
            // Default init all new elements
            if (init) {
                char* t_data = static_cast<char*>(t->getDataPtr());
                for (unsigned int i = 0; i < count; ++i) {
                    memcpy(t_data + i * variable_size, v.second.default_value, variable_size);
                }
            }
            _data->emplace(v.first, std::move(t));
        } else {
            // Need to resize the variables vector
            it->second->resize(count);
            // Default init all new elements
            if (init) {
                char* t_data = static_cast<char*>(it->second->getDataPtr());
                for (unsigned int i = _capacity; i < count; ++i) {
                    memcpy(t_data + i * variable_size, v.second.default_value, variable_size);
                }
            }
        }
    }
    _capacity = count;
}
void AgentVector::swap(AgentVector& other) noexcept {
    std::swap(_data, other._data);
    std::swap(_capacity, other._capacity);
    std::swap(_size, other._size);
    std::swap(agent, other.agent);
}

bool AgentVector::operator==(const AgentVector& other) const {
    if (_size != other._size)
        return false;
    if (*agent != *other.agent)
        return false;
    for (const auto& v : agent->variables) {
        const auto it_a = _data->find(v.first);
        const auto it_b = other._data->find(v.first);
        const char* data_a = static_cast<const char*>(it_a->second->getReadOnlyDataPtr());
        const char* data_b = static_cast<const char*>(it_b->second->getReadOnlyDataPtr());
        for (size_type i = 0; i < _size; ++i)
            if (data_a[i] != data_b[i])
                return false;
    }
    return true;
}
bool AgentVector::operator!=(const AgentVector& other) const {
    return !((*this) == other);
}


bool AgentVector::matchesAgentType(const AgentData& other) const { return *agent == other; }
bool AgentVector::matchesAgentType(const AgentDescription& other) const { return *agent == *other.agent; }
std::type_index AgentVector::getVariableType(const std::string &variable_name) const {
    const auto &it = agent->variables.find(variable_name);
    if (it == agent->variables.end()) {
        THROW InvalidAgentVar("Agent '%s' does not contain variable with name '%s', "
            "in AgentVector::getVariableType()\n",
            agent->name.c_str(), variable_name.c_str());
    }
    return it->second.type;
}
const VariableMap& AgentVector::getVariableMetaData() const {
    return agent->variables;
}
std::string AgentVector::getInitialState() const {
    return agent->initial_state;
}

AgentVector::Agent AgentVector::iterator::operator*() const {
    return Agent(_agent, _data, _pos);
}
AgentVector::CAgent AgentVector::const_iterator::operator*() const {
    return CAgent(_agent, _data, _pos);
}
AgentVector::Agent AgentVector::reverse_iterator::operator*() const {
    return Agent(_agent, _data, _pos);
}
AgentVector::CAgent AgentVector::const_reverse_iterator::operator*() const {
    return CAgent(_agent, _data, _pos);
}
