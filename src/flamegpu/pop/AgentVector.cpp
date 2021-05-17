#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentVector_Agent.h"

// @todo - this shouldn't be required anymore?
#ifdef max
#undef max  // Unclear where this definition is leaking from
#endif

namespace flamegpu {

const float AgentVector::RESIZE_FACTOR = 1.5f;

AgentVector::AgentVector(const AgentDescription& agent_desc, size_type count)
    : AgentVector(*agent_desc.agent, count) { }
AgentVector::AgentVector(const AgentData& agent_desc, size_type count)
    : agent(agent_desc.clone())
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
    insert(0, other.begin(), other.end());
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
    internal_resize(other.size(), false);
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
    if (pos >= _size) {
        _requireLength();
        if (pos >= _size) {
            THROW OutOfBoundsException("pos (%u) exceeds length of vector (%u) in AgentVector::at()", pos, _size);
        }
    }
    // Return the agent instance
    return Agent(this, agent, _data, pos);
}
AgentVector::CAgent AgentVector::at(size_type pos) const {
    if (pos >= _size) {
        _requireLength();
        if (pos >= _size) {
            THROW OutOfBoundsException("pos (%u) exceeds length of vector (%u) in AgentVector::at()", pos, _size);
        }
    }
    return CAgent(const_cast<AgentVector*>(this), agent, _data, pos);
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
    _requireLength();
    return at(_size - 1);
}
AgentVector::CAgent AgentVector::back() const {
    _requireLength();
    return at(_size - 1);
}

void* AgentVector::data(const std::string& variable_name) {
    if (!variable_name.empty() && variable_name[0] == '_') {
        THROW ReservedName("Agent variable names that begin with '_' are reserved for internal usage and cannot be changed directly, "
            "in AgentVector::data().");
    }
    // Is variable name found
    const auto& var = agent->variables.find(variable_name);
    if (var == agent->variables.end()) {
        THROW InvalidAgentVar("Variable with name '%s' was not found in agent '%s', "
            "in AgentVector::data().",
            variable_name.c_str(), agent->name.c_str());
    }
    // Does the map have a vector
    const auto& map_it = _data->find(variable_name);
    if (map_it != _data->end()) {
        _requireLength();
        _require(variable_name);
        _changedAfter(variable_name, 0);
        return map_it->second->getDataPtr();
    }
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
    if (map_it != _data->end()) {
        _requireLength();
        _require(variable_name);
        return map_it->second->getDataPtr();
    }
    return nullptr;
}

AgentVector::iterator AgentVector::begin() noexcept {
    _requireLength();
    return iterator(this, agent, _data, 0);
}
AgentVector::const_iterator AgentVector::begin() const noexcept {
    _requireLength();
    return const_iterator(const_cast<AgentVector*>(this), agent, _data, 0);
}
AgentVector::const_iterator AgentVector::cbegin() const noexcept {
    _requireLength();
    return const_iterator(const_cast<AgentVector*>(this), agent, _data, 0);
}
AgentVector::iterator AgentVector::end() noexcept {
    return iterator(this, agent, _data, _size);
}
AgentVector::const_iterator AgentVector::end() const noexcept {
    return const_iterator(const_cast<AgentVector*>(this), agent, _data, _size);
}
AgentVector::const_iterator AgentVector::cend() const noexcept {
    return const_iterator(const_cast<AgentVector*>(this), agent, _data, _size);
}
AgentVector::reverse_iterator AgentVector::rbegin() noexcept {
    _requireLength();
    return reverse_iterator(this, agent, _data, _size - 1);
}
AgentVector::const_reverse_iterator AgentVector::rbegin() const noexcept {
    _requireLength();
    return const_reverse_iterator(const_cast<AgentVector*>(this), agent, _data, _size-1);
}
AgentVector::const_reverse_iterator AgentVector::crbegin() const noexcept {
    _requireLength();
    return const_reverse_iterator(const_cast<AgentVector*>(this), agent, _data, _size-1);
}
AgentVector::reverse_iterator AgentVector::rend() noexcept {
    return reverse_iterator(this, agent, _data, std::numeric_limits<size_type>::max());
}
AgentVector::const_reverse_iterator AgentVector::rend() const noexcept {
    return const_reverse_iterator(const_cast<AgentVector*>(this), agent, _data, std::numeric_limits<size_type>::max());
}
AgentVector::const_reverse_iterator AgentVector::crend() const noexcept {
    return const_reverse_iterator(const_cast<AgentVector*>(this), agent, _data, std::numeric_limits<size_type>::max());
}

bool AgentVector::empty() const {
    _requireLength();
    return _size == 0;
}
AgentVector::size_type AgentVector::size() const {
    _requireLength();
    return _size;
}
AgentVector::size_type AgentVector::max_size() { return std::numeric_limits<size_type>::max() - 1; }
void AgentVector::reserve(size_type new_cap) {
    if (new_cap > _capacity) {
        internal_resize(new_cap, true);
    }
}
AgentVector::size_type AgentVector::capacity() const { return _capacity; }
void AgentVector::shrink_to_fit() {
    _requireLength();
    if (_size > _capacity) {
        internal_resize(_size, false);
    }
}
void AgentVector::clear() {
    _requireLength();
    // Re initialise all variables
    if (_capacity) {
        init(0, _size);
    }
    _size = 0;
    _erase(0, _size);
}

#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4127)
// Suppress condition expression is constant
// The constant condition can be made constexpr in future with C++17
#endif
void AgentVector::resetAllIDs() {
    _require(ID_VARIABLE_NAME);
    const auto it = _data->find(ID_VARIABLE_NAME);
    if (it != _data->end()) {
        constexpr id_t DEFAULT_VALUE = ID_NOT_SET;
        id_t* t_data = static_cast<id_t*>(it->second->getDataPtr());
        if (DEFAULT_VALUE == 0) {
            memset(t_data, 0, _size * sizeof(id_t));
        } else {
            for (unsigned int i = 0; i < _size; ++i) {
                memcpy(t_data + i, &DEFAULT_VALUE, sizeof(id_t));
            }
        }
    } else {
        THROW InvalidOperation("Agent '%s' is missing internal ID variable, "
            "in AgentVector::resetAllIDs()\n",
            agent->name.c_str());
    }
    if (_size) {
        // Mark all indices as changed (there isn't currently a single fn for this)
        _changed(ID_VARIABLE_NAME, 0);
        _changedAfter(ID_VARIABLE_NAME, 0);
    }
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
void AgentVector::init(size_type first, size_type last) {
    if (first >= last) {
        THROW InvalidOperation("Last (%u) must exceed first(%u), "
          "in AgentVector::init()\n",
          last, first);
    } else if (last > _capacity) {
        THROW OutOfBoundsException("Last (%u) exceeds capacity (%u) in AgentVector::init(), "
            "in AgentVector::init()\n",
          last, _capacity);
    }
    // Re initialise all variables
    for (const auto& v : agent->variables) {
        const auto it = _data->find(v.first);
        const size_t variable_size = v.second.type_size * v.second.elements;
        char* t_data = static_cast<char*>(it->second->getDataPtr());
        for (unsigned int i = first; i < last; ++i) {
            memcpy(t_data + i * variable_size, v.second.default_value, variable_size);
        }
    }
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
        return iterator(this, agent, _data, pos);
    // Confirm they are for the same agent type
    if (value._agent != agent && *value._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
            "in AgentVector::push_back().\n",
            value._agent->name.c_str(), agent->name.c_str());
    }
    // Expand capacity if required
    {
        _requireLength();
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + count > new_capacity) {
            new_capacity = static_cast<size_type>(new_capacity * RESIZE_FACTOR) + 1;
        }
        internal_resize(new_capacity, true);
    }
    // If we are not appending, ensure we have upto date device data
    if (pos < _size)
      _requireAll();
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
    // Notify subclasses
    _insert(pos, count);
    // Return iterator to first inserted item
    return iterator(this, agent, _data, insert_index);
}
AgentVector::iterator AgentVector::insert(const_iterator pos, size_type count, const Agent& value) {
    return insert(pos._pos, count, value);
}
AgentVector::iterator AgentVector::insert(size_type pos, size_type count, const Agent& value) {
    if (count == 0)
        return iterator(this, agent, _data, pos);
    // Confirm they are for the same agent type
    if (value._agent != agent && *value._agent != *agent) {
        THROW InvalidAgent("Agent description mismatch, '%' provided, '%' required, "
            "in AgentVector::push_back().\n",
            value._agent->name.c_str(), agent->name.c_str());
    }
    // Expand capacity if required
    {
        _requireLength();
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + count > new_capacity) {
            new_capacity = static_cast<size_type>(new_capacity * RESIZE_FACTOR) + 1;
        }
        internal_resize(new_capacity, true);
    }
    // If we are not appending, ensure we have upto date device data
    if (pos < _size)
        _requireAll();
    // Get first index;
    const size_type insert_index = pos;
    // Fix each variable
    auto value_data = value._data.lock();
    if (!value_data) {
        THROW ExpiredWeakPtr("The AgentVector which owns the passed AgentVector::Agent has been deallocated, "
            "in AgentVector::insert().\n");
    }
    const id_t ID_DEFAULT = ID_NOT_SET;
    for (const auto& v : agent->variables) {
        const auto it = _data->find(v.first);
        char* t_data = static_cast<char*>(it->second->getDataPtr());
        const size_t variable_size = v.second.type_size * v.second.elements;
        // Move all items after this index backwards count places
        for (unsigned int i = _size - 1; i >= insert_index; --i) {
            // Copy items individually, incase the src and destination overlap
            memcpy(t_data + (i + count) * variable_size, t_data + i * variable_size, variable_size);
        }
        // Copy across item data, ID has a special case, where it is default init instead of being copied
        if (v.first == ID_VARIABLE_NAME) {
            if (v.second.elements != 1 || v.second.type != std::type_index(typeid(id_t))) {
                THROW InvalidOperation("Agent's internal ID variable is not type %s[1], "
                        "in AgentVector::insert()\n", std::type_index(typeid(id_t)).name());
            }
            for (unsigned int i = insert_index; i < insert_index + count; ++i) {
                memcpy(t_data + i * variable_size, &ID_DEFAULT, sizeof(id_t));
            }
        } else {
            const auto other_it = value_data->find(v.first);
            char* src_data = static_cast<char*>(other_it->second->getDataPtr());
            for (unsigned int i = insert_index; i < insert_index + count; ++i) {
                memcpy(t_data + i * variable_size, src_data + value.index * variable_size, variable_size);
            }
        }
    }
    // Increase size
    _size += count;
    // Notify subclasses
    _insert(pos, count);
    // Return iterator to first inserted item
    return iterator(this, agent, _data, insert_index);
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
        return iterator(this, agent, _data, last);
    // Get first index;
    const size_type first_remove_index = first < last ? first : last;
    const size_type first_move_index = first < last ? last : first;
    const size_type erase_count = first_move_index - first_remove_index;
    _requireLength();
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
    // If we are not erasing from the end, ensure we have upto date device data
    if (first_move_index != _size)
        _requireAll();
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
    }
    // Initialise newly empty variables
    init(first_empty_index, last_empty_index);
    // Decrease size
    _size -= erase_count;
    // Notify subclasses
    _erase(first_remove_index, erase_count);
    // Return iterator following the last removed element
    return iterator(this, agent, _data, first_remove_index + 1);
}
void AgentVector::push_back(const AgentInstance& value) {
    insert(cend(), value);
}
void AgentVector::push_back() {
    // Expand capacity if required
    {
        _requireLength();
        size_type new_capacity = _capacity;
        assert((_capacity * RESIZE_FACTOR) + 1 > _capacity);
        while (_size + 1 > new_capacity) {
            new_capacity = static_cast<size_type>(_capacity * RESIZE_FACTOR) + 1;
        }
        internal_resize(new_capacity, true);
    }
    // Notify subclasses & increase size
    _insert(_size++, 1);
}
void AgentVector::pop_back() {
    _requireLength();
    if (_size) {
        --_size;
        // Reset removed item to default value
        for (const auto& v : agent->variables) {
            const auto it = _data->find(v.first);
            const size_t variable_size = v.second.type_size * v.second.elements;
            char* t_data = static_cast<char*>(it->second->getDataPtr());
            memcpy(t_data + _size * variable_size, v.second.default_value, variable_size);
        }
        // Notify subclasses
        _erase(_size, 1);
    }
}
void AgentVector::resize(size_type count) {
    _requireLength();
    const size_type old_size = _size;
    internal_resize(count, true);
    _size = count;
    // Notify subclasses
    if (count > old_size) {
        _insert(old_size, count - old_size);
    } else if (count < old_size) {
        _erase(count, old_size - count);
    }
}
void AgentVector::internal_resize(size_type count, bool init) {
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
        }
    }
    size_type old_capacity = _capacity;
    _capacity = count;
    // Default init all new elements
    if (init && count > old_capacity) {
        this->init(old_capacity, _capacity);
    }
}
void AgentVector::swap(AgentVector& other) noexcept {
    std::swap(_data, other._data);
    std::swap(_capacity, other._capacity);
    std::swap(_size, other._size);
    std::swap(agent, other.agent);
}

bool AgentVector::operator==(const AgentVector& other) const {
    _requireLength();
    if (_size != other._size)
        return false;
    if (*agent != *other.agent)
        return false;
    _requireAll();
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
    return Agent(_parent, _agent, _data, _pos);
}
AgentVector::CAgent AgentVector::const_iterator::operator*() const {
    return CAgent(_parent, _agent, _data, _pos);
}
AgentVector::Agent AgentVector::reverse_iterator::operator*() const {
    return Agent(_parent, _agent, _data, _pos);
}
AgentVector::CAgent AgentVector::const_reverse_iterator::operator*() const {
    return CAgent(_parent, _agent, _data, _pos);
}

}  // namespace flamegpu
