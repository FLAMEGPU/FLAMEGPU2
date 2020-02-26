#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_NEW_AGENT_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_NEW_AGENT_API_H_

#include <unordered_map>
#include <string>

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
                "in NewAgentStorage.setVariable()\n",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.setVariable()\n",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        memcpy(data + var->second.offset, &val, var->second.len);
    }
    template<typename T>
    T getVariable(const std::string &var_name) const {
        const auto &var = offsets.vars.find(var_name);
        if (var == offsets.vars.end()) {
            THROW InvalidAgentVar("Variable '%s' not found, "
                "in NewAgentStorage.getVariable()\n",
                var_name.c_str());
        }
        const auto t_type = std::type_index(typeid(T));
        if (var->second.type != std::type_index(typeid(T))) {
            THROW InvalidVarType("Variable '%s' has type '%s, incorrect  type '%s' was requested, "
                "in NewAgentStorage.getVariable()\n",
                var_name.c_str(), var->second.type.name(), t_type.name());
        }
        return *reinterpret_cast<T*>(data + var->second.offset);
    }
    /**
     * Used by CUDAAgentModel::processHostAgentCreation() which needs raw access to the data buffer
     */
    friend class CUDAAgentModel;
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
        s->setVariable<T>(var_name, val);
    }
    /**
     * Returns a varaiable within the new agent
     */
    template<typename T>
    T getVariable(const std::string &var_name) const {
        return s->getVariable<T>(var_name);
    }

 private:
    // Can't use reference here, makes it non-assignable
    NewAgentStorage *s;
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_NEW_AGENT_API_H_
