#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_NEW_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_NEW_AGENT_API_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "flamegpu/model/Variable.h"

/**
* This struct holds a map of how memory for a compact representation of some unknown vars needs to look
*/
struct VarOffsetStruct {
    struct OffsetLen {
        const ptrdiff_t offset;
        const size_t len;
        const std::type_index type;
        OffsetLen(const ptrdiff_t &_offset, const size_t &_len, const std::type_index _type)
            : offset(_offset)
            , len(_len)
            , type(_type) { }
    };
    std::unordered_map<std::string, OffsetLen> vars;
    const size_t totalSize;
    char *const default_data;
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
    explicit VarOffsetStruct(const VarOffsetStruct &other)
        : vars(other.vars)
        , totalSize(other.totalSize)
        , default_data(reinterpret_cast<char*>(malloc(totalSize))) {
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
* This struct provides a compact smemory store for storing generic variables in a single struct
*/
struct NewAgentStorage {
    explicit NewAgentStorage(const VarOffsetStruct &v)
        : data(reinterpret_cast<char*>(malloc(v.totalSize)))
        , offsets(v) {
        memcpy(data, offsets.default_data, offsets.totalSize);
    }
    NewAgentStorage(const NewAgentStorage &other)
        : data(reinterpret_cast<char*>(malloc(other.offsets.totalSize)))
        , offsets(other.offsets) {
        memcpy(data, other.data, offsets.totalSize);
    }
    ~NewAgentStorage() {
        free(data);
    }
    template<typename T>
    void setVariable(const std::string &var_name, const T &val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(T)) {
            THROW InvalidAgentVar("This method is not suitable for agent array variables, "
                " variable '%s' was passed, "
                "in NewAgentStorage::setVariable().",
                var_name.c_str());
        }
        memcpy(data + var->second.offset, &val, var->second.len);
    }
    template<typename T, unsigned int N>
    void setVariable(const std::string &var_name, const std::array<T, N> &val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str());
        }
        // if (var.second.len == 1 || N == 1) {
        //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::setVariable().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(T) * N) {
            THROW InvalidVarArrayLen("Variable '%s' is an array with %u elements, incorrect array of length %u was provided, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str(), var->second.len / sizeof(T), N);
        }
        memcpy(data + var->second.offset, val.data(), var->second.len);
    }
    template<typename T>
    void setVariable(const std::string &var_name, const unsigned int &index, const T &val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str());
        }
        // if (var.second.len == 1) {
        //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::setVariable().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len < sizeof(T) * (index + 1)) {
            THROW OutOfRangeVarArray("Variable '%s' is an array with %u elements, index %u is out of range, "
                "in NewAgentStorage.setVariable().",
                var_name.c_str(), var->second.len / sizeof(T), index);
        }
        memcpy(data + var->second.offset + (index * sizeof(T)), &val, sizeof(T));
    }
#ifdef SWIG
    template<typename T>
    void setVariableArray(const std::string &var_name, const std::vector<T> &val) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.setVariableArray().",
                var_name.c_str());
        }
        // if (var.second.len == 1 || N == 1) {
        //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::setVariableArray().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.setVariableArray().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(T) * val.size()) {
            THROW InvalidVarArrayLen("Variable '%s' is an array with %u elements, incorrect array of length %u was provided, "
                "in NewAgentStorage.setVariableArray().",
                var_name.c_str(), var->second.len / sizeof(T), val.size());
        }
        memcpy(data + var->second.offset, val.data(), var->second.len);
    }
#endif
    template<typename T>
    T getVariable(const std::string &var_name) const {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.getVariable()",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(T)) {
            THROW InvalidAgentVar("This method is not suitable for agent array variables, "
                " variable '%s' was passed, "
                "in NewAgentStorage::getVariable().",
                var_name.c_str());
        }
        return *reinterpret_cast<T*>(data + var->second.offset);
    }
    template<typename T, unsigned int N>
    std::array<T, N> getVariable(const std::string &var_name) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str());
        }
        // if (var.second.len == 1 || N == 1) {
        //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::getVariable().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len != sizeof(T) * N) {
            THROW InvalidVarArrayLen("Variable '%s' is an array with %u elements, incorrect array of length %u was specified, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str(), var->second.len / sizeof(T), N);
        }
        std::array<T, N> rtn;
        memcpy(rtn.data(), data + var->second.offset, var->second.len);
        return rtn;
    }
    template<typename T>
    T getVariable(const std::string &var_name, const unsigned int &index) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str());
        }
        // if (var.second.len == 1) {
        //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
        //         "in NewAgentStorage::getVariable().",
        //         var_name.c_str());
        // }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        if (var->second.len < sizeof(T) * (index + 1)) {
            THROW OutOfRangeVarArray("Variable '%s' is an array with %u elements, index %u is out of range, "
                "in NewAgentStorage.getVariable().",
                var_name.c_str(), var->second.len / sizeof(T), index);
        }
        return *reinterpret_cast<T*>(data + var->second.offset + (index * sizeof(T)));
    }
#ifdef SWIG
    template<typename T>
    std::vector<T> getVariableArray(const std::string &var_name) {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.getVariableArray().",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.getVariableArray().",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        const size_t elements = var->second.len / sizeof(T);
        std::vector<T> rtn(elements);
        memcpy(rtn.data(), data + var->second.offset, var->second.len);
        return rtn;
    }
#endif
    /**
     * Used by CUDASimulation::processHostAgentCreation() which needs raw access to the data buffer
     */
    friend class CUDASimulation;
    friend class CUDAEnsemble;
 private:
    char *const data;
    const VarOffsetStruct &offsets;
};

/**
 * This is the main API class used by a user for creating new agents on the host
 */
class FLAMEGPU_HOST_NEW_AGENT_API {
 public:
    /**
     * Assigns a new agent it's storage
     */
    explicit FLAMEGPU_HOST_NEW_AGENT_API(NewAgentStorage &_s)
        : s(&_s) { }
    /**
     * Copy Constructor
     * This does not duplicate the agent, they both point to the same data, it updates the pointed to agent data
     */
    FLAMEGPU_HOST_NEW_AGENT_API(const FLAMEGPU_HOST_NEW_AGENT_API &hna)
        : s(hna.s) { }
    /**
     * Assignment Operator
     * This does not duplicate the agent, it updates the pointed to agent data
     */
    FLAMEGPU_HOST_NEW_AGENT_API& operator=(const FLAMEGPU_HOST_NEW_AGENT_API &hna) {
        s = hna.s;
        return *this;
    }

    /**
     * Updates a varaiable within the new agent
     */
    template<typename T>
    void setVariable(const std::string &var_name, const T &val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in FLAMEGPU_HOST_NEW_AGENT_API::setVariable().");
        }
        s->setVariable<T>(var_name, val);
    }
    template<typename T, unsigned int N>
    void setVariable(const std::string &var_name, const std::array<T, N> &val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in FLAMEGPU_HOST_NEW_AGENT_API::setVariable().");
        }
        s->setVariable<T, N>(var_name, val);
    }
    template<typename T>
    void setVariable(const std::string &var_name, const unsigned int &index, const T &val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in FLAMEGPU_HOST_NEW_AGENT_API::setVariable().");
        }
        s->setVariable<T>(var_name, index, val);
    }
#ifdef SWIG
    template<typename T>
    void setVariableArray(const std::string &var_name, const std::vector<T> &val) {
        if (!var_name.empty() && var_name[0] == '_') {
            THROW ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
                "in FLAMEGPU_HOST_NEW_AGENT_API::setVariable().");
        }
        s->setVariableArray<T>(var_name, val);
    }
#endif
    /**
     * Returns a varaiable within the new agent
     */
    template<typename T>
    T getVariable(const std::string &var_name) const {
        return s->getVariable<T>(var_name);
    }
    template<typename T, unsigned int N>
    std::array<T, N> getVariable(const std::string &var_name) {
        return s->getVariable<T, N>(var_name);
    }
    template<typename T>
    T getVariable(const std::string &var_name, const unsigned int &index) {
        return s->getVariable<T>(var_name, index);
    }
#ifdef SWIG
    template<typename T>
    std::vector<T> getVariableArray(const std::string &var_name) {
        return s->getVariableArray<T>(var_name);
    }
#endif

 private:
    // Can't use reference here, makes it non-assignable
    NewAgentStorage *s;
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_NEW_AGENT_API_H_
