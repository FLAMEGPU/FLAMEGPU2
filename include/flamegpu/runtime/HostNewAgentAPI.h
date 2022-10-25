#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTNEWAGENTAPI_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTNEWAGENTAPI_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "flamegpu/model/Variable.h"
#include "flamegpu/defines.h"
#include "flamegpu/util/type_decode.h"

namespace flamegpu {

/**
* This struct holds a map of how memory for a compact representation of some unknown vars needs to look
*/
struct VarOffsetStruct {
    /**
     * Tuple containing the offset, length and type index of a variable
     */
    struct OffsetLen {
        const ptrdiff_t offset;
        const size_t len;
        const std::type_index type;
        /**
         * Constructor
         * @param _offset Offset of the variable within the buffer
         * @param _len Length of the variables data within the buffer
         * @param _type Type of the variable's base type (does not account for whether it's an array)
         */
        OffsetLen(const ptrdiff_t &_offset, const size_t &_len, const std::type_index _type)
            : offset(_offset)
            , len(_len)
            , type(_type) { }
        /**
         * Equality operator, returns true if all 3 components match
         */
        bool operator==(const OffsetLen& other) const {
            return offset == other.offset && len == other.len && type == other.type;
        }
    };
    std::unordered_map<std::string, OffsetLen> vars;
    const size_t totalSize;
    char *const default_data;
    /**
     * Construct a new VarOffsetStruct from a map of variable metadata
     * @param vmap Map of variable metadata to construct for
     */
    explicit VarOffsetStruct(const VariableMap &vmap)
        : totalSize(buildVars(vmap))
        , default_data(reinterpret_cast<char*>(malloc(totalSize))) {
        // Fill default_data with default struct
        memset(default_data, 0, totalSize);
        for (const auto &a : vmap) {
            const auto &b = vars.at(a.first);
            if (a.second.default_value)
                memcpy(default_data + b.offset, a.second.default_value, b.len);
        }
    }
    /**
     * Copy constructor
     * @param other Item to be copied
     */
    explicit VarOffsetStruct(const VarOffsetStruct &other)
        : vars(other.vars)
        , totalSize(other.totalSize)
        , default_data(reinterpret_cast<char*>(malloc(other.totalSize))) {
        memcpy(default_data, other.default_data, totalSize);
    }
    ~VarOffsetStruct() {
        free(default_data);
    }

 private:
    size_t buildVars(const VariableMap &vmap) {
        size_t _totalAgentSize = 0;
        for (const auto &a : vmap) {
            vars.emplace(a.first, OffsetLen(_totalAgentSize, a.second.type_size * a.second.elements, a.second.type));
            _totalAgentSize += a.second.type_size * a.second.elements;
        }
        return _totalAgentSize;
    }
};
/**
* This struct provides a compact memory store for storing generic variables in a single struct
*/
struct NewAgentStorage {
    explicit NewAgentStorage(const VarOffsetStruct &v, id_t id)
        : data(reinterpret_cast<char*>(malloc(v.totalSize)))
        , offsets(v) {
        memcpy(data, offsets.default_data, offsets.totalSize);
        // Overwrite _id value
        const auto& var = offsets.vars.find(ID_VARIABLE_NAME);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidOperation("Internal agent ID variable was not found, "
                "in NewAgentStorage.NewAgentStorage().");
        }
        // Don't bother checking type/len
        memcpy(data + var->second.offset, &id, sizeof(id_t));
    }
    /**
     * New agent storage cannot be copied, as it requires a unique ID
     */
    NewAgentStorage(const NewAgentStorage &other) = delete;
    /**
     * New agent storage can be moved
     */
    NewAgentStorage(NewAgentStorage &&other) noexcept
      : data(other.data)
      , offsets(other.offsets) {
      other.data = nullptr;
    }
    /**
     * Assigning new agent storage copies all items except for internal members (variables that begin with _, such as _id)
     */
    NewAgentStorage& operator=(const NewAgentStorage& hna) {
        if (offsets.vars == hna.offsets.vars) {
            // Iterate and copy all vars individually, skip those marked as internal
            for (const auto &off : offsets.vars) {
                if (off.first[0] != '_') {
                    memcpy(this->data + off.second.offset, hna.data + off.second.offset, off.second.len);
                }
            }
        } else {
            THROW exception::InvalidArgument("Attempting to assign data from agent of different type, in NewAgentStorage::operator=()\n");
        }
        return *this;
    }
    ~NewAgentStorage() {
        if (data)
            free(data);
    }
    template<typename T>
    void setVariable(const std::string &var_name, const T val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s', incorrect  type '%s' was requested, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t) {
            THROW exception::InvalidAgentVar("This method is not suitable for agent array variables, "
                " variable '%s' was passed, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str());
        }
        memcpy(data + var->second.offset, &val, var->second.len);
    }
    template<typename T, unsigned int N = 0>
    void setVariable(const std::string &var_name, const unsigned int index, const T val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str());
        }
        if (N && N != var->second.len / sizeof(typename type_decode<T>::type_t)) {
            THROW exception::InvalidAgentVar("Agent variable '%s' length mismatch %u != %u, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str(), N, var->second.len / sizeof(typename type_decode<T>::type_t));
        }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len < (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t) * (index + 1)) {
            THROW exception::OutOfRangeVarArray("Variable '%s' is an array with %u elements, index %u is out of range, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str(), var->second.len / (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t), index);
        }
        memcpy(data + var->second.offset + (index * sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t), &val, sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t);
    }
#ifndef SWIG
    template<typename T, unsigned int N>
    void setVariable(const std::string &var_name, const std::array<T, N> &val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str());
        }
        // if (var.second.len == 1 || N == 1) {
        //     THROW exception::InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::setVariable().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t * N) {
            THROW exception::InvalidVarArrayLen("Variable '%s' is an array with %u elements, incorrect array of length %u was provided, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str(), var->second.len / (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t), N);
        }
        memcpy(data + var->second.offset, val.data(), var->second.len);
    }
#else
    template<typename T>
    void setVariableArray(const std::string &var_name, const std::vector<T> &val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::setVariableArray().",
                var_name.c_str());
        }
        // if (var.second.len == 1 || N == 1) {
        //     THROW exception::InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::setVariableArray().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::setVariableArray().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t * val.size()) {
            THROW exception::InvalidVarArrayLen("Variable '%s' is an array with %u elements, incorrect array of length %u was provided, "
                "in NewAgentStorage::setVariableArray().",
                var_name.c_str(), var->second.len / (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t), val.size());
        }
        memcpy(data + var->second.offset, val.data(), var->second.len);
    }
#endif
    template<typename T>
    T getVariable(const std::string &var_name) const {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::getVariable()",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t) {
            THROW exception::InvalidAgentVar("This method is not suitable for agent array variables, "
                " variable '%s' was passed, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str());
        }
        return *reinterpret_cast<T*>(data + var->second.offset);
    }
    template<typename T, unsigned int N = 0>
    T getVariable(const std::string &var_name, const unsigned int index) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str());
        }
        if (N && N != var->second.len / sizeof(typename type_decode<T>::type_t)) {
            THROW exception::InvalidAgentVar("Agent variable '%s' length mismatch %u != %u, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str(), N, var->second.len / sizeof(typename type_decode<T>::type_t));
        }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len < sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t * (index + 1)) {
            THROW exception::OutOfRangeVarArray("Variable '%s' is an array with %u elements, index %u is out of range, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str(), var->second.len / (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t), index);
        }
        return *reinterpret_cast<T*>(data + var->second.offset + (index * sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t));
    }
#ifndef SWIG
    template<typename T, unsigned int N>
    std::array<T, N> getVariable(const std::string &var_name) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str());
        }
        // if (var.second.len == 1 || N == 1) {
        //     THROW exception::InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::getVariable().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t * N) {
            THROW exception::InvalidVarArrayLen("Variable '%s' is an array with %u elements, incorrect array of length %u was specified, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str(), var->second.len / (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t), N);
        }
        std::array<T, N> rtn;
        memcpy(rtn.data(), data + var->second.offset, var->second.len);
        return rtn;
    }
#else
    template<typename T>
    std::vector<T> getVariableArray(const std::string &var_name) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW exception::InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage::getVariableArray().",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(typename type_decode<T>::type_t));
        if (var->second.type != t_type) {
            THROW exception::InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage::getVariableArray().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len % (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t) != 0) {
            THROW exception::InvalidVarType("Variable '%s' has length (%llu) is not divisible by vector length (%u), "
                "in NewAgentStorage::getVariableArray().",
                var_name.c_str(), var->second.len / sizeof(typename type_decode<T>::type_t), type_decode<T>::len_t);
        }
        const size_t elements = var->second.len / (sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t);
        std::vector<T> rtn(elements);
        memcpy(rtn.data(), data + var->second.offset, var->second.len);
        return rtn;
    }
#endif
    /**
     * Used by CUDASimulation::processHostAgentCreation() which needs raw access to the data buffer
     */
    friend class CUDASimulation;
    /**
     * Used by DeviceAgentVector which needs raw access to the data buffer if a dependency requires it
     */
    friend class DeviceAgentVector_impl;
 private:
    char *data;
    const VarOffsetStruct &offsets;
};

/**
 * This is the main API class used by a user for creating new agents on the host
 */
class HostNewAgentAPI {
 public:
    /**
     * Assigns a new agent it's storage
     */
    explicit HostNewAgentAPI(NewAgentStorage &_s)
        : s(&_s) { }
    /**
     * Copy Constructor
     * This does not duplicate the agent, they both point to the same data, it updates the pointed to agent data
     */
    HostNewAgentAPI(const HostNewAgentAPI &hna)
        : s(hna.s) { }
    /**
     * Assignment Operator
     * This copies (non-internal) agent variable data from hna
     * @throws exception::InvalidArgument If hna is of a different agent type (has a different internal memory layout)
     */
    HostNewAgentAPI& operator=(const HostNewAgentAPI &hna) {
        if (&hna != this)
            *s = *hna.s;
        return *this;
    }

    /**
     * Updates a variable within the new agent
     */
    template<typename T>
    void setVariable(const std::string &var_name, const T val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW exception::ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in HostNewAgentAPI::setVariable().");
        }
        s->setVariable<T>(var_name, val);
    }
    template<typename T, unsigned int N = 0>
    void setVariable(const std::string &var_name, const unsigned int index, const T val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW exception::ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in HostNewAgentAPI::setVariable().");
        }
        s->setVariable<T, N>(var_name, index, val);
    }
#ifndef SWIG
    template<typename T, unsigned int N>
    void setVariable(const std::string& var_name, const std::array<T, N>& val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW exception::ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in HostNewAgentAPI::setVariable().");
        }
        s->setVariable<T, N>(var_name, val);
    }
#else
    template<typename T>
    void setVariableArray(const std::string &var_name, const std::vector<T> &val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW exception::ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in HostNewAgentAPI::setVariable().");
        }
        s->setVariableArray<T>(var_name, val);
    }
#endif
    /**
     * Returns a variable within the new agent
     */
    template<typename T>
    T getVariable(const std::string &var_name) const {
        return s->getVariable<T>(var_name);
    }
    template<typename T, unsigned int N = 0>
    T getVariable(const std::string &var_name, const unsigned int index) {
        return s->getVariable<T, N>(var_name, index);
    }
#ifndef SWIG
    template<typename T, unsigned int N>
    std::array<T, N> getVariable(const std::string& var_name) {
        return s->getVariable<T, N>(var_name);
    }
#else
    template<typename T>
    std::vector<T> getVariableArray(const std::string &var_name) {
        return s->getVariableArray<T>(var_name);
    }
#endif
    /**
     * Returns the agent's unique ID
     */
    id_t getID() const {
        try {
            return s->getVariable<id_t>(ID_VARIABLE_NAME);
        } catch (...) {
            // Rewrite all exceptions
            THROW exception::UnknownInternalError("Internal Error: Unable to read internal ID variable, in HostNewAgentAPI::getID()\n");
        }
    }

 private:
    // Can't use reference here, makes it non-assignable
    NewAgentStorage *s;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTNEWAGENTAPI_H_
