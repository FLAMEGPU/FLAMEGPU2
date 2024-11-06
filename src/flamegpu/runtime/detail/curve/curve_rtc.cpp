#include <sstream>
#include <set>
#include <string>

#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/simulation/detail/EnvironmentManager.cuh"
#include "flamegpu/detail/cuda.cuh"

// jitify include for demangle
#ifdef _MSC_VER
#pragma warning(push, 2)
#include "jitify/jitify.hpp"
#pragma warning(pop)
#else
#include "jitify/jitify.hpp"
#endif

namespace flamegpu {
namespace detail {
namespace curve {


const char* CurveRTCHost::curve_rtc_dynamic_h_template = R"###(dynamic/curve_rtc_dynamic.h
#line 1 "$FILENAME"
#ifndef CURVE_RTC_DYNAMIC_H_
#define CURVE_RTC_DYNAMIC_H_

#include "flamegpu/exception/FLAMEGPUDeviceException.cuh"
#include "flamegpu/detail/type_decode.h"
#include "flamegpu/runtime/detail/curve/Curve.cuh"
#include "flamegpu/util/dstring.h"

namespace flamegpu {

template <unsigned int N, unsigned int I> struct StringCompare {
    __device__ inline static bool strings_equal_loop(const char(&a)[N], const char(&b)[N]) {
        return a[N - I] == b[N - I] && StringCompare<N, I - 1>::strings_equal_loop(a, b);
    }
};

template <unsigned int N> struct StringCompare<N, 1> {
    __device__ inline static bool strings_equal_loop(const char(&a)[N], const char(&b)[N]) {
        return a[0] == b[0];
    }
};

template <unsigned int N>
__device__ bool strings_equal(const char(&a)[N], const char(&b)[N]) {
    return StringCompare<N, N>::strings_equal_loop(a, b);
}

template <unsigned int N, unsigned int M>
__device__ bool strings_equal(const char(&a)[N], const char(&b)[M]) {
    return false;
}

namespace detail {
namespace curve {

/**
 * Dynamically generated version of Curve without hashing
 * Both environment data, and curve variable ptrs are stored in this buffer
 * Order: Env Data, Agent, MessageOut, MessageIn, NewAgent
 * EnvData size must be a multiple of 8 bytes
 */
$DYNAMIC_VARIABLES

class DeviceCurve {
    public:
    static const int UNKNOWN_VARIABLE = -1;
    static const unsigned int UNKNOWN_GRAPH = 0xFFFFFFFF;  // UINT_MAX

    typedef int                      Variable;
    typedef unsigned int             VariableHash;
    typedef unsigned int             NamespaceHash;
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable(const char(&name)[N], unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable(const char(&name)[N], unsigned int index);
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getAgentVariable_ldg(const char(&name)[N], unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static T getMessageVariable_ldg(const char(&name)[N], unsigned int index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable(const char(&name)[M], unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getMessageArrayVariable(const char(&name)[M], unsigned int variable_index, unsigned int array_index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getAgentArrayVariable_ldg(const char(&name)[M], unsigned int variable_index, unsigned int array_index);    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getMessageArrayVariable_ldg(const char(&name)[M], unsigned int variable_index, unsigned int array_index);
    
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setAgentVariable(const char(&name)[N], T variable, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setMessageVariable(const char(&name)[N], T variable, unsigned int index);
    template <typename T, unsigned int N>
    __device__ __forceinline__ static void setNewAgentVariable(const char(&name)[N], T variable, unsigned int index);
    
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setAgentArrayVariable(const char(&name)[M], T variable, unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setMessageArrayVariable(const char(&name)[M], T variable, unsigned int variable_index, unsigned int array_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static void setNewAgentArrayVariable(const char(&name)[M], T variable, unsigned int variable_index, unsigned int array_index);

    __device__ __forceinline__ static unsigned int *getEnvironmentDirectedGraphPBM(VariableHash graphHash);
    __device__ __forceinline__ static unsigned int *getEnvironmentDirectedGraphIPBM(VariableHash graphHash);
    __device__ __forceinline__ static unsigned int *getEnvironmentDirectedGraphIPBMEdges(VariableHash graphHash);
    
    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getEnvironmentDirectedGraphVertexProperty(VariableHash graphHash, const char(&propertyName)[M], unsigned int vertex_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getEnvironmentDirectedGraphVertexArrayProperty(VariableHash graphHash, const char(&propertyName)[M], unsigned int vertex_index, unsigned int array_index);

    template <typename T, unsigned int M>
    __device__ __forceinline__ static T getEnvironmentDirectedGraphEdgeProperty(VariableHash graphHash, const char(&propertyName)[M], unsigned int edge_index);
    template <typename T, unsigned int N, unsigned int M>
    __device__ __forceinline__ static T getEnvironmentDirectedGraphEdgeArrayProperty(VariableHash graphHash, const char(&propertyName)[M], unsigned int edge_index, unsigned int array_index);
    
    template <unsigned int M>
    __device__ __forceinline__ static VariableHash getGraphHash(const char(&graphName)[M]);

    template <unsigned int M>
    __device__ __forceinline__ static unsigned int getVariableCount(const char(&variableName)[M], const VariableHash namespace_hash);

    __device__ __forceinline__ static bool isAgent(const char* agent_name);
    __device__ __forceinline__ static bool isState(const char* agent_state);
};

template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getAgentVariable(const char (&name)[N], unsigned int index) {
$DYNAMIC_GETAGENTVARIABLE_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getMessageVariable(const char (&name)[N], unsigned int index) {
$DYNAMIC_GETMESSAGEVARIABLE_IMPL
}

template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getEnvironmentDirectedGraphVertexProperty(VariableHash graphHash, const char(&name)[N], unsigned int index) {
$DYNAMIC_GETDIRECTEDGRAPHVERTEXPROPERTY_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getEnvironmentDirectedGraphEdgeProperty(VariableHash graphHash, const char(&name)[N], unsigned int index) {
$DYNAMIC_GETDIRECTEDGRAPHEDGEPROPERTY_IMPL
}

template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getAgentVariable_ldg(const char (&name)[N], unsigned int index) {
$DYNAMIC_GETAGENTVARIABLE_LDG_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ T DeviceCurve::getMessageVariable_ldg(const char (&name)[N], unsigned int index) {
$DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentArrayVariable(const char(&name)[M], unsigned int index, unsigned int array_index) {
$DYNAMIC_GETAGENTARRAYVARIABLE_IMPL
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageArrayVariable(const char(&name)[M], unsigned int index, unsigned int array_index) {
$DYNAMIC_GETMESSAGEARRAYVARIABLE_IMPL
}
    

__device__ __forceinline__ unsigned int *DeviceCurve::getEnvironmentDirectedGraphPBM(VariableHash graphHash) {
$DYNAMIC_GETDIRECTEDGRAPHPBM_IMPL
}
__device__ __forceinline__ unsigned int *DeviceCurve::getEnvironmentDirectedGraphIPBM(VariableHash graphHash) {
$DYNAMIC_GETDIRECTEDGRAPHIPBM_IMPL
}
__device__ __forceinline__ unsigned int *DeviceCurve::getEnvironmentDirectedGraphIPBMEdges(VariableHash graphHash) {
$DYNAMIC_GETDIRECTEDGRAPHIPBMEDGES_IMPL
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getEnvironmentDirectedGraphVertexArrayProperty(VariableHash graphHash, const char(&name)[M], unsigned int index, unsigned int array_index) {
$DYNAMIC_GETDIRECTEDGRAPHVERTEXARRAYPROPERTY_IMPL
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getEnvironmentDirectedGraphEdgeArrayProperty(VariableHash graphHash, const char(&name)[M], unsigned int index, unsigned int array_index) {
$DYNAMIC_GETDIRECTEDGRAPHEDGEARRAYPROPERTY_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getAgentArrayVariable_ldg(const char(&name)[M], unsigned int index, unsigned int array_index) {
$DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T DeviceCurve::getMessageArrayVariable_ldg(const char(&name)[M], unsigned int index, unsigned int array_index) {
$DYNAMIC_GETMESSAGEARRAYVARIABLE_LDG_IMPL
}

template <typename T, unsigned int N>
__device__ __forceinline__ void DeviceCurve::setAgentVariable(const char(&name)[N], T variable, unsigned int index) {
$DYNAMIC_SETAGENTVARIABLE_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ void DeviceCurve::setMessageVariable(const char(&name)[N], T variable, unsigned int index) {
$DYNAMIC_SETMESSAGEVARIABLE_IMPL
}
template <typename T, unsigned int N>
__device__ __forceinline__ void DeviceCurve::setNewAgentVariable(const char(&name)[N], T variable, unsigned int index) {
$DYNAMIC_SETNEWAGENTVARIABLE_IMPL
}

template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setAgentArrayVariable(const char(&name)[M], T variable, unsigned int index, unsigned int array_index) {
$DYNAMIC_SETAGENTARRAYVARIABLE_IMPL    
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setMessageArrayVariable(const char(&name)[M], T variable, unsigned int index, unsigned int array_index) {
$DYNAMIC_SETMESSAGEARRAYVARIABLE_IMPL    
}
template <typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ void DeviceCurve::setNewAgentArrayVariable(const char(&name)[M], T variable, unsigned int index, unsigned int array_index) {
$DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL    
}

template <unsigned int M>
__device__ __forceinline__ DeviceCurve::VariableHash DeviceCurve::getGraphHash(const char(&graphName)[M]) {
$DYNAMIC_GETGRAPHHASH_IMPL
}

template <unsigned int M>
__device__ __forceinline__ unsigned int DeviceCurve::getVariableCount(const char(&variableName)[M], const VariableHash namespace_hash) {
$DYNAMIC_GETVARIABLECOUNT_IMPL
}

__device__ __forceinline__ bool DeviceCurve::isAgent(const char* agent_name) {
    return util::dstrcmp(agent_name, "$DYNAMIC_AGENT_NAME") == 0;
}
__device__ __forceinline__ bool DeviceCurve::isState(const char* agent_state) {
    return util::dstrcmp(agent_state, "$DYNAMIC_AGENT_STATE") == 0;
}

}  // namespace curve 
}  // namespace detail 
}  // namespace flamegpu 

// has to be included after definition of curve namespace
#include "flamegpu/runtime/environment/DeviceEnvironment.cuh"
//#include "flamegpu/runtime/environment/DeviceMacroProperty.cuh"

namespace flamegpu {

template<typename T, unsigned int M>
__device__ __forceinline__ T ReadOnlyDeviceEnvironment::getProperty(const char(&name)[M]) const {
$DYNAMIC_ENV_GETVARIABLE_IMPL
}

template<typename T, unsigned int N, unsigned int M>
__device__ __forceinline__ T ReadOnlyDeviceEnvironment::getProperty(const char(&name)[M], const unsigned int index) const {
$DYNAMIC_ENV_GETARRAYVARIABLE_IMPL
}


template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int N>
__device__ __forceinline__ ReadOnlyDeviceMacroProperty<T, I, J, K, W> ReadOnlyDeviceEnvironment::getMacroProperty(const char(&name)[N]) const {
$DYNAMIC_ENV_GETREADONLYMACROPROPERTY_IMPL
}
template<typename T, unsigned int I, unsigned int J, unsigned int K, unsigned int W, unsigned int N>
__device__ __forceinline__ DeviceMacroProperty<T, I, J, K, W> DeviceEnvironment::getMacroProperty(const char(&name)[N]) const {
$DYNAMIC_ENV_GETMACROPROPERTY_IMPL
}

}  // namespace flamegpu

#endif  // CURVE_RTC_DYNAMIC_H_
)###";


CurveRTCHost::CurveRTCHost() : header(CurveRTCHost::curve_rtc_dynamic_h_template) {
}

CurveRTCHost::~CurveRTCHost() {
    gpuErrchk(flamegpu::detail::cuda::cudaFreeHost(h_data_buffer));
}

void CurveRTCHost::registerAgentVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    props.count_index = static_cast<unsigned int>(count_buffer.size()); count_buffer.push_back(0);
    if (!agent_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerAgentVariable()", variableName);
    }
}
void CurveRTCHost::registerMessageInVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    props.count_index = static_cast<unsigned int>(count_buffer.size()); count_buffer.push_back(0);
    if (!messageIn_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerMessageInVariable()", variableName);
    }
}
void CurveRTCHost::registerMessageOutVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    props.count_index = static_cast<unsigned int>(count_buffer.size()); count_buffer.push_back(0);
    if (!messageOut_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerMessageOutVariable()", variableName);
    }
}
void CurveRTCHost::registerNewAgentVariable(const char* variableName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    props.count_index = static_cast<unsigned int>(count_buffer.size()); count_buffer.push_back(0);
    if (!newAgent_variables.emplace(variableName, props).second) {
        THROW exception::UnknownInternalError("Variable '%s' is already registered, in CurveRTCHost::registerNewAgentVariable()", variableName);
    }
}
void CurveRTCHost::registerEnvironmentDirectedGraphVertexProperty(const std::string& graphName, const std::string& propertyName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    props.count_index = static_cast<unsigned int>(count_buffer.size()); count_buffer.push_back(0);
    if (!directedGraph_vertexProperties.emplace(std::pair{ graphName, propertyName}, props).second) {
        THROW exception::UnknownInternalError("Property '%s' is already registered, in CurveRTCHost::registerEnvironmentDirectedGraphVertexProperty()", propertyName.c_str());
    }
}
void CurveRTCHost::registerEnvironmentDirectedGraphEdgeProperty(const std::string& graphName, const std::string& propertyName, const char* type, size_t type_size, unsigned int elements, bool read, bool write) {
    RTCVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.read = read;
    props.write = write;
    props.elements = elements;
    props.type_size = type_size;
    props.count_index = static_cast<unsigned int>(count_buffer.size()); count_buffer.push_back(0);
    if (!directedGraph_edgeProperties.emplace(std::pair{ graphName, propertyName }, props).second) {
        THROW exception::UnknownInternalError("Property '%s' is already registered, in CurveRTCHost::registerEnvironmentDirectedGraphEdgeProperty()", propertyName.c_str());
    }
}

void CurveRTCHost::unregisterAgentVariable(const char* variableName) {
    auto i = agent_variables.find(variableName);
    if (i != agent_variables.end()) {
        agent_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterAgentVariable()", variableName);
    }
}
void CurveRTCHost::unregisterMessageOutVariable(const char* variableName) {
    auto i = messageOut_variables.find(variableName);
    if (i != messageOut_variables.end()) {
        messageOut_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterMessageOutVariable()", variableName);
    }
}
void CurveRTCHost::unregisterMessageInVariable(const char* variableName) {
    auto i = messageIn_variables.find(variableName);
    if (i != messageIn_variables.end()) {
        messageIn_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterMessageInVariable()", variableName);
    }
}
void CurveRTCHost::unregisterNewAgentVariable(const char* variableName) {
    auto i = newAgent_variables.find(variableName);
    if (i != newAgent_variables.end()) {
        newAgent_variables.erase(variableName);
    } else {
        THROW exception::UnknownInternalError("Variable '%s' not found when removing variable, in CurveRTCHost::unregisterNewAgentVariable()", variableName);
    }
}
void CurveRTCHost::unregisterEnvironmentDirectedGraphVertexProperty(const std::string& graphName, const std::string& propertyName) {
    auto i = directedGraph_vertexProperties.find({graphName, propertyName});
    if (i != directedGraph_vertexProperties.end()) {
        directedGraph_vertexProperties.erase({graphName, propertyName});
    } else {
        THROW exception::UnknownInternalError("Vertex property '%s' from graph '%s' not found when removing property, "
          "in CurveRTCHost::unregisterEnvironmentDirectedGraphVertexProperty()", propertyName.c_str(), graphName.c_str());
    }
}
void CurveRTCHost::unregisterEnvironmentDirectedGraphEdgeProperty(const std::string& graphName, const std::string& propertyName) {
    auto i = directedGraph_edgeProperties.find({graphName, propertyName});
    if (i != directedGraph_edgeProperties.end()) {
        directedGraph_edgeProperties.erase({graphName, propertyName});
    } else {
        THROW exception::UnknownInternalError("Edge property '%s' from graph '%s' not found when removing property, "
            "in CurveRTCHost::unregisterEnvironmentDirectedGraphEdgeProperty()", propertyName.c_str(), graphName.c_str());
    }
}


void* CurveRTCHost::getAgentVariableCachePtr(const char* variableName) {
    const auto i = agent_variables.find(variableName);
    if (i != agent_variables.end()) {
        if (i->second.h_data_ptr)
            return i->second.h_data_ptr;
        THROW exception::UnknownInternalError("Variable '%s' has not yet been allocated within the cache, in CurveRTCHost::getAgentVariableCachePtr()", variableName);
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getAgentVariableCachePtr()", variableName);
}
void* CurveRTCHost::getMessageOutVariableCachePtr(const char* variableName) {
    const auto i = messageOut_variables.find(variableName);
    if (i != messageOut_variables.end()) {
        if (i->second.h_data_ptr)
            return i->second.h_data_ptr;
        THROW exception::UnknownInternalError("Variable '%s' has not yet been allocated within the cache, in CurveRTCHost::getMessageOutVariableCachePtr()", variableName);
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getMessageOutVariableCachePtr()", variableName);
}
void* CurveRTCHost::getMessageInVariableCachePtr(const char* variableName) {
    const auto i = messageIn_variables.find(variableName);
    if (i != messageIn_variables.end()) {
        if (i->second.h_data_ptr)
            return i->second.h_data_ptr;
        THROW exception::UnknownInternalError("Variable '%s' has not yet been allocated within the cache, in CurveRTCHost::getMessageInVariableCachePtr()", variableName);
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getMessageInVariableCachePtr()", variableName);
}
void* CurveRTCHost::getNewAgentVariableCachePtr(const char* variableName) {
    const auto i = newAgent_variables.find(variableName);
    if (i != newAgent_variables.end()) {
        if (i->second.h_data_ptr)
            return i->second.h_data_ptr;
        THROW exception::UnknownInternalError("Variable '%s' has not yet been allocated within the cache, in CurveRTCHost::getNewAgentVariableCachePtr()", variableName);
    }
    THROW exception::UnknownInternalError("Variable '%s' not found when accessing variable, in CurveRTCHost::getNewAgentVariableCachePtr()", variableName);
}
void* CurveRTCHost::getEnvironmentDirectedGraphVertexPropertyCachePtr(const std::string& graphName, const std::string& propertyName) {
    const auto i = directedGraph_vertexProperties.find(std::pair{graphName, propertyName});
    if (i != directedGraph_vertexProperties.end()) {
        if (i->second.h_data_ptr)
            return i->second.h_data_ptr;
        THROW exception::UnknownInternalError("Vertex property '%s' from graph '%s' has not yet been allocated within the cache, in CurveRTCHost::getEnvironmentDirectedGraphVertexPropertyCachePtr()", propertyName.c_str(), graphName.c_str());
    }
    THROW exception::UnknownInternalError("Vertex property '%s' from graph '%s' not found when retrieving property, "
        "in CurveRTCHost::getEnvironmentDirectedGraphVertexPropertyCachePtr()", propertyName.c_str(), graphName.c_str());
}
void* CurveRTCHost::getEnvironmentDirectedGraphEdgePropertyCachePtr(const std::string& graphName, const std::string& propertyName) {
    const auto i = directedGraph_edgeProperties.find(std::pair{graphName, propertyName});
    if (i != directedGraph_edgeProperties.end()) {
        if (i->second.h_data_ptr)
            return i->second.h_data_ptr;
        THROW exception::UnknownInternalError("Edge property '%s' from graph '%s' has not yet been allocated within the cache, in CurveRTCHost::getEnvironmentDirectedGraphEdgePropertyCachePtr()", propertyName.c_str(), graphName.c_str());
    }
    THROW exception::UnknownInternalError("Edge property '%s' from graph '%s' not found when retrieving property, "
        "in CurveRTCHost::getEnvironmentDirectedGraphEdgePropertyCachePtr()", propertyName.c_str(), graphName.c_str());
}
void CurveRTCHost::setAgentVariableCount(const std::string& variableName, const unsigned int count) {
    const auto i = agent_variables.find(variableName);
    if (i != agent_variables.end()) {
        count_buffer[i->second.count_index] = count;
        return;
    }
    THROW exception::UnknownInternalError("Agent variable '%s' not found when retrieving property, "
        "in CurveRTCHost::setAgentVariablePropertyCount()", variableName.c_str());
}
void CurveRTCHost::setMessageOutVariableCount(const std::string& variableName, const unsigned int count) {
    const auto i = messageOut_variables.find(variableName);
    if (i != messageOut_variables.end()) {
        count_buffer[i->second.count_index] = count;
        return;
    }
    THROW exception::UnknownInternalError("Message variable '%s' not found when retrieving property, "
        "in CurveRTCHost::setMessageOutVariablePropertyCount()", variableName.c_str());
}
void CurveRTCHost::setMessageInVariableCount(const std::string& variableName, const unsigned int count) {
    const auto i = messageIn_variables.find(variableName);
    if (i != messageIn_variables.end()) {
        count_buffer[i->second.count_index] = count;
        return;
    }
    THROW exception::UnknownInternalError("Message variable '%s' not found when retrieving property, "
        "in CurveRTCHost::setMessageInVariableCount()", variableName.c_str());
}
void CurveRTCHost::setNewAgentVariableCount(const std::string& variableName, const unsigned int count) {
    const auto i = newAgent_variables.find(variableName);
    if (i != newAgent_variables.end()) {
        count_buffer[i->second.count_index] = count;
        return;
    }
    THROW exception::UnknownInternalError("Agent variable '%s' not found when retrieving property, "
        "in CurveRTCHost::setNewAgentVariableCount()", variableName.c_str());
}
void CurveRTCHost::setEnvironmentDirectedGraphVertexPropertyCount(const std::string& graphName, const std::string& propertyName, const unsigned int count) {
    const auto i = directedGraph_vertexProperties.find(std::pair{ graphName, propertyName });
    if (i != directedGraph_vertexProperties.end()) {
        count_buffer[i->second.count_index] = count;
        return;
    }
    THROW exception::UnknownInternalError("Vertex property '%s' from graph '%s' not found when retrieving property, "
        "in CurveRTCHost::setEnvironmentDirectedGraphVertexPropertyCount()", propertyName.c_str(), graphName.c_str());
}
void CurveRTCHost::setEnvironmentDirectedGraphEdgePropertyCount(const std::string& graphName, const std::string& propertyName, const unsigned int count) {
    const auto i = directedGraph_edgeProperties.find(std::pair{ graphName, propertyName });
    if (i != directedGraph_edgeProperties.end()) {
        count_buffer[i->second.count_index] = count;
        return;
    }
    THROW exception::UnknownInternalError("Edge property '%s' from graph '%s' not found when retrieving property, "
        "in CurveRTCHost::setEnvironmentDirectedGraphEdgePropertyCount()", propertyName.c_str(), graphName.c_str());
}

void CurveRTCHost::registerEnvVariable(const char* propertyName, ptrdiff_t offset, const char* type, size_t type_size, unsigned int elements) {
    RTCEnvVariableProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.elements = elements;
    props.offset = offset;
    props.type_size = type_size;
    if (!RTCEnvVariables.emplace(propertyName, props).second) {
        THROW exception::UnknownInternalError("Environment property with name '%s' is already registered, in CurveRTCHost::registerEnvVariable()", propertyName);
    }
}
void CurveRTCHost::registerAgent(const std::string &_agentName, const std::string &_agentState) {
    if (this->agentName.empty()) {
        this->agentName = _agentName;
        this->agentState = _agentState;
    } else {
        THROW exception::UnknownInternalError("Agent is already registered with name '%s' and state '%s', in CurveRTCHost::registerAgent()", this->agentName.c_str(), this->agentState.c_str());
    }
}

void CurveRTCHost::unregisterEnvVariable(const char* propertyName) {
    auto i = RTCEnvVariables.find(propertyName);
    if (i != RTCEnvVariables.end()) {
        RTCEnvVariables.erase(propertyName);
    } else {
        THROW exception::UnknownInternalError("Environment property '%s' not found when removing environment property, in CurveRTCHost::unregisterEnvVariable()", propertyName);
    }
}
void CurveRTCHost::registerEnvMacroProperty(const char* propertyName, void *d_ptr, const char* type, size_t type_size, const std::array<unsigned int, 4> &dimensions) {
    RTCEnvMacroPropertyProperties props;
    props.type = CurveRTCHost::demangle(type);
    props.dimensions = dimensions;
    props.d_ptr = d_ptr;
    props.h_data_ptr = nullptr;
    props.type_size = type_size;
    if (!RTCEnvMacroProperties.emplace(propertyName, props).second) {
        THROW exception::UnknownInternalError("Environment property with name '%s' is already registered, in CurveRTCHost::registerEnvMacroProperty()", propertyName);
    }
}

void CurveRTCHost::unregisterEnvMacroProperty(const char* propertyName) {
    auto i = RTCEnvMacroProperties.find(propertyName);
    if (i != RTCEnvMacroProperties.end()) {
        RTCEnvMacroProperties.erase(propertyName);
    } else {
        THROW exception::UnknownInternalError("Environment macro property '%s' not found when removing environment property, in CurveRTCHost::unregisterEnvMacroProperty()", propertyName);
    }
}


void CurveRTCHost::initHeaderEnvironment(const size_t env_buffer_len) {
    // Calculate size of, and generate dynamic variables buffer
    std::stringstream variables;
    data_buffer_size = env_buffer_len;
    // Fix alignment
    data_buffer_size += (data_buffer_size % sizeof(void*) != 0) ? sizeof(void*) - (data_buffer_size % sizeof(void*)) : 0;

    agent_data_offset = data_buffer_size;     data_buffer_size += agent_variables.size() * sizeof(void*);
    messageOut_data_offset = data_buffer_size;    data_buffer_size += messageOut_variables.size() * sizeof(void*);
    messageIn_data_offset = data_buffer_size;     data_buffer_size += messageIn_variables.size() * sizeof(void*);
    newAgent_data_offset = data_buffer_size;  data_buffer_size += newAgent_variables.size() * sizeof(void*);
    directedGraphVertex_data_offset = data_buffer_size;  data_buffer_size += directedGraph_vertexProperties.size() * sizeof(void*);
    directedGraphEdge_data_offset = data_buffer_size;  data_buffer_size += directedGraph_edgeProperties.size() * sizeof(void*);
    envMacro_data_offset = data_buffer_size;  data_buffer_size += RTCEnvMacroProperties.size() * sizeof(void*);
    count_data_offset = data_buffer_size;  data_buffer_size += count_buffer.size() * sizeof(unsigned int);
    variables << "__constant__  char " << getVariableSymbolName() << "[" << data_buffer_size << "];\n";
    setHeaderPlaceholder("$DYNAMIC_VARIABLES", variables.str());
    // generate Environment::get func implementation ($DYNAMIC_ENV_GETVARIABLE_IMPL)
    {
        std::stringstream getEnvVariableImpl;
        for (std::pair<std::string, RTCEnvVariableProperties> element : RTCEnvVariables) {
            RTCEnvVariableProperties props = element.second;
            {
                getEnvVariableImpl <<   "    if (strings_equal(name, \"" << element.first << "\")) {\n";
                getEnvVariableImpl <<   "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getEnvVariableImpl <<   "        if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getEnvVariableImpl <<   "            DTHROW(\"Environment property '%s' type mismatch.\\n\", name);\n";
                getEnvVariableImpl <<   "            return {};\n";
                getEnvVariableImpl <<   "        } else if(detail::type_decode<T>::len_t != " << element.second.elements << ") {\n";
                getEnvVariableImpl <<   "            DTHROW(\"Environment property '%s' length mismatch.\\n\", name);\n";
                getEnvVariableImpl <<   "            return {};\n";
                getEnvVariableImpl <<   "        }\n";
                getEnvVariableImpl <<   "#endif\n";
                getEnvVariableImpl <<   "        return *reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() <<" + " << props.offset << "));\n";
                getEnvVariableImpl <<   "    };\n";
            }
        }
        getEnvVariableImpl <<           "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getEnvVariableImpl <<           "    DTHROW(\"Environment property '%s' was not found.\\n\", name);\n";
        getEnvVariableImpl <<           "#endif\n";
        getEnvVariableImpl <<           "    return  {};\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_GETVARIABLE_IMPL", getEnvVariableImpl.str());
    }
    // generate Environment::get func implementation for array variables ($DYNAMIC_ENV_GETARRAYVARIABLE_IMPL)
    {
        std::stringstream getEnvArrayVariableImpl;
        for (std::pair<std::string, RTCEnvVariableProperties> element : RTCEnvVariables) {
            RTCEnvVariableProperties props = element.second;
            if (props.elements > 1) {
                getEnvArrayVariableImpl << "    if (strings_equal(name, \"" << element.first << "\")) {\n";
                getEnvArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getEnvArrayVariableImpl << "        const unsigned int t_index = detail::type_decode<T>::len_t * index + detail::type_decode<T>::len_t;\n";
                getEnvArrayVariableImpl << "        if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getEnvArrayVariableImpl << "            DTHROW(\"Environment array property '%s' type mismatch.\\n\", name);\n";
                getEnvArrayVariableImpl << "            return {};\n";
                getEnvArrayVariableImpl << "        } else if (detail::type_decode<T>::len_t * N != " << element.second.elements << " && N != 0) {\n";  // Special case, env array specifying length is optional as it's not actually required
                getEnvArrayVariableImpl << "            DTHROW(\"Environment array property '%s' length mismatch.\\n\", name);\n";
                getEnvArrayVariableImpl << "            return {};\n";
                getEnvArrayVariableImpl << "        } else if (t_index > " << element.second.elements << " || t_index < index) {\n";
                getEnvArrayVariableImpl << "            DTHROW(\"Environment array property '%s', index %d is out of bounds.\\n\", name, index);\n";
                getEnvArrayVariableImpl << "            return {};\n";
                getEnvArrayVariableImpl << "        }\n";
                getEnvArrayVariableImpl << "#endif\n";
                getEnvArrayVariableImpl << "        return reinterpret_cast<T*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() <<" + " << props.offset << "))[index];\n";
                getEnvArrayVariableImpl << "    };\n";
            }
        }
        getEnvArrayVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getEnvArrayVariableImpl <<         "    DTHROW(\"Environment array property '%s' was not found.\\n\", name);\n";
        getEnvArrayVariableImpl <<         "#endif\n";
        getEnvArrayVariableImpl <<         "    return {};\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_GETARRAYVARIABLE_IMPL", getEnvArrayVariableImpl.str());
    }
    // generate Environment::getMacroProperty func implementation ($DYNAMIC_ENV_GETREADONLYMACROPROPERTY_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMacroPropertyImpl;
        for (std::pair<std::string, RTCEnvMacroPropertyProperties> element : RTCEnvMacroProperties) {
            RTCEnvMacroPropertyProperties props = element.second;
            getMacroPropertyImpl << "    if (strings_equal(name, \"" << element.first << "\")) {\n";
            getMacroPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
            getMacroPropertyImpl << "        if(sizeof(T) != " << element.second.type_size << ") {\n";
            getMacroPropertyImpl << "            DTHROW(\"Environment macro property '%s' type mismatch.\\n\", name);\n";
            getMacroPropertyImpl << "        } else if (I != " << element.second.dimensions[0] << " ||\n";
            getMacroPropertyImpl << "            J != " << element.second.dimensions[1] << " |\n";
            getMacroPropertyImpl << "            K != " << element.second.dimensions[2] << " |\n";
            getMacroPropertyImpl << "            W != " << element.second.dimensions[3] << ") {\n";
            getMacroPropertyImpl << "            DTHROW(\"Environment macro property '%s' dimensions do not match (%u, %u, %u, %u) != (%u, %u, %u, %u).\\n\", name,\n";
            getMacroPropertyImpl << "                I, J, K, W, " << element.second.dimensions[0] << ", " << element.second.dimensions[1] << ", " << element.second.dimensions[2] << ", " << element.second.dimensions[3] << ");\n";
            getMacroPropertyImpl << "        } else {\n";
            getMacroPropertyImpl << "            return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(*reinterpret_cast<T**>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << envMacro_data_offset + (ct * sizeof(void*)) << "),\n";
            // Read-write flag resides in 8 bits at the end of the buffer
            getMacroPropertyImpl << "                reinterpret_cast<unsigned int*>(*reinterpret_cast<char**>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << envMacro_data_offset + (ct * sizeof(void*)) << ") + (I * J * K * W * sizeof(T))));\n";
            getMacroPropertyImpl << "        }\n";
            getMacroPropertyImpl << "#else\n";
            getMacroPropertyImpl << "        return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(*reinterpret_cast<T**>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << envMacro_data_offset + (ct * sizeof(void*)) << "));\n";
            getMacroPropertyImpl << "#endif\n";
            getMacroPropertyImpl << "    };\n";
            ++ct;
        }
        getMacroPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getMacroPropertyImpl << "    DTHROW(\"Environment macro property '%s' was not found.\\n\", name);\n";
        getMacroPropertyImpl << "    return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(nullptr, nullptr);\n";
        getMacroPropertyImpl << "#else\n";
        getMacroPropertyImpl << "    return ReadOnlyDeviceMacroProperty<T, I, J, K, W>(nullptr);\n";
        getMacroPropertyImpl << "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_GETREADONLYMACROPROPERTY_IMPL", getMacroPropertyImpl.str());
    }
    // generate Environment::getMacroProperty func implementation ($DYNAMIC_ENV_GETMACROPROPERTY_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMacroPropertyImpl;
        for (std::pair<std::string, RTCEnvMacroPropertyProperties> element : RTCEnvMacroProperties) {
            RTCEnvMacroPropertyProperties props = element.second;
            getMacroPropertyImpl << "    if (strings_equal(name, \"" << element.first << "\")) {\n";
            getMacroPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
            getMacroPropertyImpl << "        if(sizeof(T) != " << element.second.type_size << ") {\n";
            getMacroPropertyImpl << "            DTHROW(\"Environment macro property '%s' type mismatch.\\n\", name);\n";
            getMacroPropertyImpl << "        } else if (I != " << element.second.dimensions[0] << " ||\n";
            getMacroPropertyImpl << "            J != " << element.second.dimensions[1] << " |\n";
            getMacroPropertyImpl << "            K != " << element.second.dimensions[2] << " |\n";
            getMacroPropertyImpl << "            W != " << element.second.dimensions[3] << ") {\n";
            getMacroPropertyImpl << "            DTHROW(\"Environment macro property '%s' dimensions do not match (%u, %u, %u, %u) != (%u, %u, %u, %u).\\n\", name,\n";
            getMacroPropertyImpl << "                I, J, K, W, " << element.second.dimensions[0] << ", " << element.second.dimensions[1] << ", " << element.second.dimensions[2] << ", " << element.second.dimensions[3] << ");\n";
            getMacroPropertyImpl << "        } else {\n";
            getMacroPropertyImpl << "            return DeviceMacroProperty<T, I, J, K, W>(*reinterpret_cast<T**>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << envMacro_data_offset + (ct * sizeof(void*)) << "),\n";
            // Read-write flag resides in 8 bits at the end of the buffer
            getMacroPropertyImpl << "                reinterpret_cast<unsigned int*>(*reinterpret_cast<char**>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << envMacro_data_offset + (ct * sizeof(void*)) << ") + (I * J * K * W * sizeof(T))));\n";
            getMacroPropertyImpl << "        }\n";
            getMacroPropertyImpl << "#else\n";
            getMacroPropertyImpl << "        return DeviceMacroProperty<T, I, J, K, W>(*reinterpret_cast<T**>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << envMacro_data_offset + (ct * sizeof(void*)) << "));\n";
            getMacroPropertyImpl << "#endif\n";
            getMacroPropertyImpl << "    };\n";
            ++ct;
        }
        getMacroPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getMacroPropertyImpl << "    DTHROW(\"Environment macro property '%s' was not found.\\n\", name);\n";
        getMacroPropertyImpl << "    return DeviceMacroProperty<T, I, J, K, W>(nullptr, nullptr);\n";
        getMacroPropertyImpl << "#else\n";
        getMacroPropertyImpl << "    return DeviceMacroProperty<T, I, J, K, W>(nullptr);\n";
        getMacroPropertyImpl << "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_ENV_GETMACROPROPERTY_IMPL", getMacroPropertyImpl.str());
    }
}
void CurveRTCHost::initHeaderSetters() {
    // generate setAgentVariable func implementation ($DYNAMIC_SETAGENTVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setAgentVariableImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write) {
                setAgentVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setAgentVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                setAgentVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                setAgentVariableImpl << "                  DTHROW(\"Agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setAgentVariableImpl << "                  return;\n";
                setAgentVariableImpl << "              } else if(detail::type_decode<T>::len_t != " << element.second.elements << ") {\n";
                setAgentVariableImpl << "                  DTHROW(\"Agent variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setAgentVariableImpl << "                  return;\n";
                setAgentVariableImpl << "              }\n";
                setAgentVariableImpl << "#endif\n";
                setAgentVariableImpl << "              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[index] = (T) variable;\n";
                setAgentVariableImpl << "              return;\n";
                setAgentVariableImpl << "          }\n";
            } else {
                ++ct;
            }
        }
        setAgentVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        setAgentVariableImpl <<         "          DTHROW(\"Agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setAgentVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETAGENTVARIABLE_IMPL", setAgentVariableImpl.str());
    }
    // generate setMessageVariable func implementation ($DYNAMIC_SETMESSAGEVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setMessageVariableImpl;
        for (const auto &element : messageOut_variables) {
            RTCVariableProperties props = element.second;
            if (props.write) {
                setMessageVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setMessageVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                setMessageVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                setMessageVariableImpl << "                  DTHROW(\"Message variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setMessageVariableImpl << "                  return;\n";
                setMessageVariableImpl << "              } else if(detail::type_decode<T>::len_t != " << element.second.elements << ") {\n";
                setMessageVariableImpl << "                  DTHROW(\"Message variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setMessageVariableImpl << "                  return;\n";
                setMessageVariableImpl << "              }\n";
                setMessageVariableImpl << "#endif\n";
                setMessageVariableImpl << "              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << messageOut_data_offset + (ct++ * sizeof(void*)) << ")))[index] = (T) variable;\n";
                setMessageVariableImpl << "              return;\n";
                setMessageVariableImpl << "          }\n";
            } else {
                ++ct;
            }
        }
        setMessageVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        setMessageVariableImpl <<         "          DTHROW(\"Message variable '%s' was not found during setVariable().\\n\", name);\n";
        setMessageVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETMESSAGEVARIABLE_IMPL", setMessageVariableImpl.str());
    }
    // generate setNewAgentVariable func implementation ($DYNAMIC_SETNEWAGENTVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setNewAgentVariableImpl;
        for (const auto &element : newAgent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write) {
                setNewAgentVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setNewAgentVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                setNewAgentVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                setNewAgentVariableImpl << "                  DTHROW(\"New agent variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setNewAgentVariableImpl << "                  return;\n";
                setNewAgentVariableImpl << "              } else if(detail::type_decode<T>::len_t != " << element.second.elements << ") {\n";
                setNewAgentVariableImpl << "                  DTHROW(\"New agent variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setNewAgentVariableImpl << "                  return;\n";
                setNewAgentVariableImpl << "              }\n";
                setNewAgentVariableImpl << "#endif\n";
                setNewAgentVariableImpl << "              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << newAgent_data_offset + (ct++ * sizeof(void*)) << ")))[index] = (T) variable;\n";
                setNewAgentVariableImpl << "              return;\n";
                setNewAgentVariableImpl << "          }\n";
            } else {
                ++ct;
            }
        }
        setNewAgentVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        setNewAgentVariableImpl <<         "          DTHROW(\"New agent variable '%s' was not found during setVariable().\\n\", name);\n";
        setNewAgentVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETNEWAGENTVARIABLE_IMPL", setNewAgentVariableImpl.str());
    }
    // generate setAgentArrayVariable func implementation ($DYNAMIC_SETAGENTARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setAgentArrayVariableImpl;
        if (!agent_variables.empty())
            setAgentArrayVariableImpl <<             "    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;\n";
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setAgentArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                setAgentArrayVariableImpl << "              const unsigned int t_index = detail::type_decode<T>::len_t * array_index + detail::type_decode<T>::len_t;\n";
                setAgentArrayVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                setAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setAgentArrayVariableImpl << "                  return;\n";
                setAgentArrayVariableImpl << "              } else if (detail::type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                setAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setAgentArrayVariableImpl << "                  return;\n";
                setAgentArrayVariableImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                setAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during setVariable().\\n\", name, array_index);\n";
                setAgentArrayVariableImpl << "                  return;\n";
                setAgentArrayVariableImpl << "              }\n";
                setAgentArrayVariableImpl << "#endif\n";
                setAgentArrayVariableImpl << "              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[i] = (T) variable;\n";
                setAgentArrayVariableImpl << "              return;\n";
                setAgentArrayVariableImpl << "          }\n";
            } else {
                ++ct;
            }
        }
        setAgentArrayVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        setAgentArrayVariableImpl <<         "          DTHROW(\"Agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setAgentArrayVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETAGENTARRAYVARIABLE_IMPL", setAgentArrayVariableImpl.str());
    }
    // generate setMessageArrayVariable func implementation ($DYNAMIC_SETMESSAGEARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setMessageArrayVariableImpl;
        if (!messageOut_variables.empty())
            setMessageArrayVariableImpl << "    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;\n";
        for (const auto& element : messageOut_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setMessageArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setMessageArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                setMessageArrayVariableImpl << "              const unsigned int t_index = detail::type_decode<T>::len_t * array_index + detail::type_decode<T>::len_t;\n";
                setMessageArrayVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                setMessageArrayVariableImpl << "                  DTHROW(\"Message array variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setMessageArrayVariableImpl << "                  return;\n";
                setMessageArrayVariableImpl << "              } else if (detail::type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                setMessageArrayVariableImpl << "                  DTHROW(\"Message array variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setMessageArrayVariableImpl << "                  return;\n";
                setMessageArrayVariableImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                setMessageArrayVariableImpl << "                  DTHROW(\"Message array variable '%s', index %d is out of bounds during setVariable().\\n\", name, array_index);\n";
                setMessageArrayVariableImpl << "                  return;\n";
                setMessageArrayVariableImpl << "              }\n";
                setMessageArrayVariableImpl << "#endif\n";
                setMessageArrayVariableImpl << "              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << messageOut_data_offset + (ct++ * sizeof(void*)) << ")))[i] = (T) variable;\n";
                setMessageArrayVariableImpl << "              return;\n";
                setMessageArrayVariableImpl << "          }\n";
            } else {
                ++ct;
            }
        }
        setMessageArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        setMessageArrayVariableImpl << "          DTHROW(\"Message array variable '%s' was not found during setVariable().\\n\", name);\n";
        setMessageArrayVariableImpl << "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETMESSAGEARRAYVARIABLE_IMPL", setMessageArrayVariableImpl.str());
    }
    // generate setNewAgentArrayVariable func implementation ($DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream setNewAgentArrayVariableImpl;
        if (!newAgent_variables.empty())
            setNewAgentArrayVariableImpl << "    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;\n";
        for (const auto &element : newAgent_variables) {
            RTCVariableProperties props = element.second;
            if (props.write && props.elements > 1) {
                setNewAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                setNewAgentArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                setNewAgentArrayVariableImpl << "              const unsigned int t_index = detail::type_decode<T>::len_t * array_index + detail::type_decode<T>::len_t;\n";
                setNewAgentArrayVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                setNewAgentArrayVariableImpl << "                  DTHROW(\"New agent array variable '%s' type mismatch during setVariable().\\n\", name);\n";
                setNewAgentArrayVariableImpl << "                  return;\n";
                setNewAgentArrayVariableImpl << "              } else if (detail::type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                setNewAgentArrayVariableImpl << "                  DTHROW(\"New agent array variable '%s' length mismatch during setVariable().\\n\", name);\n";
                setNewAgentArrayVariableImpl << "                  return;\n";
                setNewAgentArrayVariableImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                setNewAgentArrayVariableImpl << "                  DTHROW(\"New agent array variable '%s', index %d is out of bounds during setVariable().\\n\", name, array_index);\n";
                setNewAgentArrayVariableImpl << "                  return;\n";
                setNewAgentArrayVariableImpl << "              }\n";
                setNewAgentArrayVariableImpl << "#endif\n";
                setNewAgentArrayVariableImpl << "              (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << newAgent_data_offset + (ct++ * sizeof(void*)) << ")))[i] = (T) variable;\n";
                setNewAgentArrayVariableImpl << "              return;\n";
                setNewAgentArrayVariableImpl << "          }\n";
            } else {
                ++ct;
            }
        }
        setNewAgentArrayVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        setNewAgentArrayVariableImpl <<         "          DTHROW(\"New agent array variable '%s' was not found during setVariable().\\n\", name);\n";
        setNewAgentArrayVariableImpl <<         "#endif\n";
        setHeaderPlaceholder("$DYNAMIC_SETNEWAGENTARRAYVARIABLE_IMPL", setNewAgentArrayVariableImpl.str());
    }
}
void CurveRTCHost::initHeaderGetters() {
    // generate getAgentVariable func implementation ($DYNAMIC_GETAGENTVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentVariableImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read) {
                getAgentVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getAgentVariableImpl << "                if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getAgentVariableImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentVariableImpl << "                    return {};\n";
                getAgentVariableImpl << "                } else if(detail::type_decode<T>::len_t != " << element.second.elements << ") {\n";
                getAgentVariableImpl << "                    DTHROW(\"Agent variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getAgentVariableImpl << "                    return {};\n";
                getAgentVariableImpl << "                }\n";
                getAgentVariableImpl << "#endif\n";
                getAgentVariableImpl << "                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[index];\n";
                getAgentVariableImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getAgentVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getAgentVariableImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentVariableImpl <<         "#endif\n";
        getAgentVariableImpl <<         "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTVARIABLE_IMPL", getAgentVariableImpl.str());
    }
    // generate getMessageVariable func implementation ($DYNAMIC_GETMESSAGEVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMessageVariableImpl;
        for (const auto &element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read) {
                getMessageVariableImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getMessageVariableImpl << "                if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getMessageVariableImpl << "                    DTHROW(\"Message variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageVariableImpl << "                    return {};\n";
                getMessageVariableImpl << "                } else if(detail::type_decode<T>::len_t != " << element.second.elements << ") {\n";
                getMessageVariableImpl << "                    DTHROW(\"Message variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getMessageVariableImpl << "                    return {};\n";
                getMessageVariableImpl << "                }\n";
                getMessageVariableImpl << "#endif\n";
                getMessageVariableImpl << "                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << messageIn_data_offset + (ct++ * sizeof(void*)) << ")))[index];\n";
                getMessageVariableImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getMessageVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getMessageVariableImpl <<         "            DTHROW(\"Message variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageVariableImpl <<         "#endif\n";
        getMessageVariableImpl <<         "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEVARIABLE_IMPL", getMessageVariableImpl.str());
    }
    // getEnvironmentDirectedGraphPBM
    {
        size_t ct = 0;
        std::stringstream getDirectedGraphPBMImpl;
        for (const auto& element : directedGraph_vertexProperties) {
            RTCVariableProperties props = element.second;
            if (props.read  && element.first.second == GRAPH_VERTEX_PBM_VARIABLE_NAME) {
                getDirectedGraphPBMImpl << "            if (graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getDirectedGraphPBMImpl << "                return (*static_cast<unsigned int**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphVertex_data_offset + (ct++ * sizeof(void*)) << ")));\n";
                getDirectedGraphPBMImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getDirectedGraphPBMImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getDirectedGraphPBMImpl << "            DTHROW(\"Directed graph PBM was not found during getEnvironmentDirectedGraphPBM().\\n\");\n";
        getDirectedGraphPBMImpl << "#endif\n";
        getDirectedGraphPBMImpl << "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHPBM_IMPL", getDirectedGraphPBMImpl.str());
    }
    // getEnvironmentDirectedGraphIPBM
    {
        size_t ct = 0;
        std::stringstream getDirectedGraphIPBMImpl;
        for (const auto& element : directedGraph_vertexProperties) {
            RTCVariableProperties props = element.second;
            if (props.read && element.first.second == GRAPH_VERTEX_IPBM_VARIABLE_NAME) {
                getDirectedGraphIPBMImpl << "            if (graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getDirectedGraphIPBMImpl << "                return (*static_cast<unsigned int**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphVertex_data_offset + (ct++ * sizeof(void*)) << ")));\n";
                getDirectedGraphIPBMImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getDirectedGraphIPBMImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getDirectedGraphIPBMImpl << "            DTHROW(\"Directed graph IPBM was not found during getEnvironmentDirectedGraphIPBM().\\n\");\n";
        getDirectedGraphIPBMImpl << "#endif\n";
        getDirectedGraphIPBMImpl << "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHIPBM_IMPL", getDirectedGraphIPBMImpl.str());
    }
    // getEnvironmentDirectedGraphIPBMEdges
    {
        size_t ct = 0;
        std::stringstream getDirectedGraphIPBMEdgesImpl;
        for (const auto& element : directedGraph_vertexProperties) {
            RTCVariableProperties props = element.second;
            if (props.read && element.first.second == GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME) {
                getDirectedGraphIPBMEdgesImpl << "            if (graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getDirectedGraphIPBMEdgesImpl << "                return (*static_cast<unsigned int**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphVertex_data_offset + (ct++ * sizeof(void*)) << ")));\n";
                getDirectedGraphIPBMEdgesImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getDirectedGraphIPBMEdgesImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getDirectedGraphIPBMEdgesImpl << "            DTHROW(\"Directed graph IPBM edge list was not found during getEnvironmentDirectedGraphIPBMEdges().\\n\");\n";
        getDirectedGraphIPBMEdgesImpl << "#endif\n";
        getDirectedGraphIPBMEdgesImpl << "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHIPBMEDGES_IMPL", getDirectedGraphIPBMEdgesImpl.str());
    }
    // generate getEnvironmentDirectedGraphVertexProperty func implementation ($DYNAMIC_GETDIRECTEDGRAPHVERTEXPROPERTY_IMPL)
    {
        size_t ct = 0;
        std::stringstream getGraphVertexPropertyImpl;
        for (const auto& element : directedGraph_vertexProperties) {
            RTCVariableProperties props = element.second;
            if (props.read) {
                getGraphVertexPropertyImpl << "            if (strings_equal(name, \"" << element.first.second << "\") && graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getGraphVertexPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getGraphVertexPropertyImpl << "                if(sizeof(type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getGraphVertexPropertyImpl << "                    DTHROW(\"Directed graph vertex property '%s' type mismatch during getProperty().\\n\", name);\n";
                getGraphVertexPropertyImpl << "                    return {};\n";
                getGraphVertexPropertyImpl << "                } else if(type_decode<T>::len_t != " << element.second.elements << ") {\n";
                getGraphVertexPropertyImpl << "                    DTHROW(\"MDirected graph vertex property '%s' length mismatch during getProperty().\\n\", name);\n";
                getGraphVertexPropertyImpl << "                    return {};\n";
                getGraphVertexPropertyImpl << "                }\n";
                getGraphVertexPropertyImpl << "#endif\n";
                getGraphVertexPropertyImpl << "                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphVertex_data_offset + (ct++ * sizeof(void*)) << ")))[index];\n";
                getGraphVertexPropertyImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getGraphVertexPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getGraphVertexPropertyImpl << "            DTHROW(\"Directed graph vertex property '%s' was not found during getProperty().\\n\", name);\n";
        getGraphVertexPropertyImpl << "#endif\n";
        getGraphVertexPropertyImpl << "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHVERTEXPROPERTY_IMPL", getGraphVertexPropertyImpl.str());
    }
    // generate getEnvironmentDirectedGraphEdgeProperty func implementation ($DYNAMIC_GETDIRECTEDGRAPHEDGEPROPERTY_IMPL)
    {
        size_t ct = 0;
        std::stringstream getGraphEdgePropertyImpl;
        for (const auto& element : directedGraph_edgeProperties) {
            RTCVariableProperties props = element.second;
            if (props.read) {
                getGraphEdgePropertyImpl << "            if (strings_equal(name, \"" << element.first.second << "\") && graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getGraphEdgePropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getGraphEdgePropertyImpl << "                if(sizeof(type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getGraphEdgePropertyImpl << "                    DTHROW(\"Directed graph edge property '%s' type mismatch during getProperty().\\n\", name);\n";
                getGraphEdgePropertyImpl << "                    return {};\n";
                getGraphEdgePropertyImpl << "                } else if(type_decode<T>::len_t != " << element.second.elements << ") {\n";
                getGraphEdgePropertyImpl << "                    DTHROW(\"MDirected graph edge property '%s' length mismatch during getProperty().\\n\", name);\n";
                getGraphEdgePropertyImpl << "                    return {};\n";
                getGraphEdgePropertyImpl << "                }\n";
                getGraphEdgePropertyImpl << "#endif\n";
                getGraphEdgePropertyImpl << "                return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphEdge_data_offset + (ct++ * sizeof(void*)) << ")))[index];\n";
                getGraphEdgePropertyImpl << "            }\n";
            } else {
                ++ct;
            }
        }
        getGraphEdgePropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getGraphEdgePropertyImpl << "            DTHROW(\"Directed graph edge property '%s' was not found during getProperty().\\n\", name);\n";
        getGraphEdgePropertyImpl << "#endif\n";
        getGraphEdgePropertyImpl << "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHEDGEPROPERTY_IMPL", getGraphEdgePropertyImpl.str());
    }
    // generate getAgentVariable_ldg func implementation ($DYNAMIC_GETAGENTVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentVariableLDGImpl;
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {  // GLM does not support __ldg() so should not use this
                getAgentVariableLDGImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentVariableLDGImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getAgentVariableLDGImpl << "                if(sizeof(T) != " << element.second.type_size * element.second.elements << ") {\n";
                getAgentVariableLDGImpl << "                    DTHROW(\"Agent variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentVariableLDGImpl << "                    return {};\n";
                getAgentVariableLDGImpl << "                }\n";
                getAgentVariableLDGImpl << "#endif\n";
                getAgentVariableLDGImpl << "                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << agent_data_offset + (ct * sizeof(void*)) << "))) + index);\n";
                getAgentVariableLDGImpl << "            }\n";
                ++ct;  // Prev was part of the return line, but don't want confusion
            } else {
                ++ct;
            }
        }
        getAgentVariableLDGImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getAgentVariableLDGImpl <<         "            DTHROW(\"Agent variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentVariableLDGImpl <<         "#endif\n";
        getAgentVariableLDGImpl <<         "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTVARIABLE_LDG_IMPL", getAgentVariableLDGImpl.str());
    }
    // generate getMessageVariable_ldg func implementation ($DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMessageVariableLDGImpl;
        for (const auto &element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements == 1) {  // GLM does not support __ldg() so should not use this
                getMessageVariableLDGImpl << "            if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageVariableLDGImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getMessageVariableLDGImpl << "                if(sizeof(T) != " << element.second.type_size << ") {\n";
                getMessageVariableLDGImpl << "                    DTHROW(\"Message variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageVariableLDGImpl << "                    return {};\n";
                getMessageVariableLDGImpl << "                }\n";
                getMessageVariableLDGImpl << "#endif\n";
                getMessageVariableLDGImpl << "                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << messageIn_data_offset + (ct * sizeof(void*)) << "))) + index);\n";
                getMessageVariableLDGImpl << "            }\n";
                ++ct;  // Prev was part of the return line, but don't want confusion
            } else {
                ++ct;
            }
        }
        getMessageVariableLDGImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getMessageVariableLDGImpl <<         "            DTHROW(\"Message variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageVariableLDGImpl <<         "#endif\n";
        getMessageVariableLDGImpl <<         "            return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEVARIABLE_LDG_IMPL", getMessageVariableLDGImpl.str());
    }
    // generate getAgentArrayVariable func implementation ($DYNAMIC_GETAGENTARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentArrayVariableImpl;
        if (!agent_variables.empty())
            getAgentArrayVariableImpl << "    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;\n";
        for (const auto &element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getAgentArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getAgentArrayVariableImpl << "              const unsigned int t_index = detail::type_decode<T>::len_t * array_index + detail::type_decode<T>::len_t;\n";
                getAgentArrayVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableImpl << "                  return {};\n";
                getAgentArrayVariableImpl << "              } else if (detail::type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                getAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableImpl << "                  return {};\n";
                getAgentArrayVariableImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                getAgentArrayVariableImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getAgentArrayVariableImpl << "                  return {};\n";
                getAgentArrayVariableImpl << "              }\n";
                getAgentArrayVariableImpl << "#endif\n";
                getAgentArrayVariableImpl << "              return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << agent_data_offset + (ct++ * sizeof(void*)) << ")))[i];\n";
                getAgentArrayVariableImpl << "           };\n";
            } else {
                ++ct;
            }
        }
        getAgentArrayVariableImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getAgentArrayVariableImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentArrayVariableImpl <<         "#endif\n";
        getAgentArrayVariableImpl <<         "           return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTARRAYVARIABLE_IMPL", getAgentArrayVariableImpl.str());
    }
    // generate getMessageArrayVariable func implementation ($DYNAMIC_GETMESSAGEARRAYVARIABLE_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMessageArrayVariableImpl;
        if (!messageIn_variables.empty())
            getMessageArrayVariableImpl << "    const size_t i = (index * detail::type_decode<T>::len_t * N) + detail::type_decode<T>::len_t * array_index;\n";
        for (const auto& element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getMessageArrayVariableImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getMessageArrayVariableImpl << "              const unsigned int t_index = detail::type_decode<T>::len_t * array_index + detail::type_decode<T>::len_t;\n";
                getMessageArrayVariableImpl << "              if(sizeof(detail::type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getMessageArrayVariableImpl << "                  DTHROW(\"Message array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageArrayVariableImpl << "                  return {};\n";
                getMessageArrayVariableImpl << "              } else if (detail::type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                getMessageArrayVariableImpl << "                  DTHROW(\"Message array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getMessageArrayVariableImpl << "                  return {};\n";
                getMessageArrayVariableImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                getMessageArrayVariableImpl << "                  DTHROW(\"Message array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getMessageArrayVariableImpl << "                  return {};\n";
                getMessageArrayVariableImpl << "              }\n";
                getMessageArrayVariableImpl << "#endif\n";
                getMessageArrayVariableImpl << "              return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << messageIn_data_offset + (ct++ * sizeof(void*)) << ")))[i];\n";
                getMessageArrayVariableImpl << "           };\n";
            } else {
                ++ct;
            }
        }
        getMessageArrayVariableImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getMessageArrayVariableImpl << "           DTHROW(\"Message array variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageArrayVariableImpl << "#endif\n";
        getMessageArrayVariableImpl << "           return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEARRAYVARIABLE_IMPL", getMessageArrayVariableImpl.str());
    }
    // generate getEnvironmentDirectedGraphVertexArrayProperty func implementation ($DYNAMIC_GETDIRECTEDGRAPHVERTEXARRAYPROPERTY_IMPL)
    {
        size_t ct = 0;
        std::stringstream getGraphVertexArrayPropertyImpl;
        if (!directedGraph_vertexProperties.empty())
            getGraphVertexArrayPropertyImpl << "    const size_t i = (index * type_decode<T>::len_t * N) + type_decode<T>::len_t * array_index;\n";
        for (const auto& element : directedGraph_vertexProperties) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getGraphVertexArrayPropertyImpl << "          if (strings_equal(name, \"" << element.first.second << "\") && graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getGraphVertexArrayPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getGraphVertexArrayPropertyImpl << "              const unsigned int t_index = type_decode<T>::len_t * array_index + type_decode<T>::len_t;\n";
                getGraphVertexArrayPropertyImpl << "              if(sizeof(type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getGraphVertexArrayPropertyImpl << "                  DTHROW(\"Directed graph vertex array property '%s' type mismatch during getProperty().\\n\", name);\n";
                getGraphVertexArrayPropertyImpl << "                  return {};\n";
                getGraphVertexArrayPropertyImpl << "              } else if (type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                getGraphVertexArrayPropertyImpl << "                  DTHROW(\"Directed graph vertex array property '%s' length mismatch during getProperty().\\n\", name);\n";
                getGraphVertexArrayPropertyImpl << "                  return {};\n";
                getGraphVertexArrayPropertyImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                getGraphVertexArrayPropertyImpl << "                  DTHROW(\"Directed graph vertex array property '%s', index %d is out of bounds during getProperty().\\n\", name, array_index);\n";
                getGraphVertexArrayPropertyImpl << "                  return {};\n";
                getGraphVertexArrayPropertyImpl << "              }\n";
                getGraphVertexArrayPropertyImpl << "#endif\n";
                getGraphVertexArrayPropertyImpl << "              return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphVertex_data_offset + (ct++ * sizeof(void*)) << ")))[i];\n";
                getGraphVertexArrayPropertyImpl << "           };\n";
            } else {
                ++ct;
            }
        }
        getGraphVertexArrayPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getGraphVertexArrayPropertyImpl << "           DTHROW(\"Directed graph vertex array property '%s' was not found during getProperty().\\n\", name);\n";
        getGraphVertexArrayPropertyImpl << "#endif\n";
        getGraphVertexArrayPropertyImpl << "           return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHVERTEXARRAYPROPERTY_IMPL", getGraphVertexArrayPropertyImpl.str());
    }
    // generate getEnvironmentDirectedGraphEdgeArrayProperty func implementation ($DYNAMIC_GETDIRECTEDGRAPHEDGEARRAYPROPERTY_IMPL)
    {
        size_t ct = 0;
        std::stringstream getGraphEdgeArrayPropertyImpl;
        if (!directedGraph_edgeProperties.empty())
            getGraphEdgeArrayPropertyImpl << "    const size_t i = (index * type_decode<T>::len_t * N) + type_decode<T>::len_t * array_index;\n";
        for (const auto& element : directedGraph_edgeProperties) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {
                getGraphEdgeArrayPropertyImpl << "          if (strings_equal(name, \"" << element.first.second << "\") && graphHash == " << Curve::variableRuntimeHash(element.first.first) << ") {\n";
                getGraphEdgeArrayPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getGraphEdgeArrayPropertyImpl << "              const unsigned int t_index = type_decode<T>::len_t * array_index + type_decode<T>::len_t;\n";
                getGraphEdgeArrayPropertyImpl << "              if(sizeof(type_decode<T>::type_t) != " << element.second.type_size << ") {\n";
                getGraphEdgeArrayPropertyImpl << "                  DTHROW(\"Directed graph edge array property '%s' type mismatch during getProperty().\\n\", name);\n";
                getGraphEdgeArrayPropertyImpl << "                  return {};\n";
                getGraphEdgeArrayPropertyImpl << "              } else if (type_decode<T>::len_t * N != " << element.second.elements << ") {\n";
                getGraphEdgeArrayPropertyImpl << "                  DTHROW(\"Directed graph edge array property '%s' length mismatch during getProperty().\\n\", name);\n";
                getGraphEdgeArrayPropertyImpl << "                  return {};\n";
                getGraphEdgeArrayPropertyImpl << "              } else if (t_index > " << element.second.elements << " || t_index < array_index) {\n";
                getGraphEdgeArrayPropertyImpl << "                  DTHROW(\"Directed graph edge array property '%s', index %d is out of bounds during getProperty().\\n\", name, array_index);\n";
                getGraphEdgeArrayPropertyImpl << "                  return {};\n";
                getGraphEdgeArrayPropertyImpl << "              }\n";
                getGraphEdgeArrayPropertyImpl << "#endif\n";
                getGraphEdgeArrayPropertyImpl << "              return (*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << directedGraphEdge_data_offset + (ct++ * sizeof(void*)) << ")))[i];\n";
                getGraphEdgeArrayPropertyImpl << "           };\n";
            } else {
                ++ct;
            }
        }
        getGraphEdgeArrayPropertyImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getGraphEdgeArrayPropertyImpl << "           DTHROW(\"Directed graph edge array property '%s' was not found during getProperty().\\n\", name);\n";
        getGraphEdgeArrayPropertyImpl << "#endif\n";
        getGraphEdgeArrayPropertyImpl << "           return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETDIRECTEDGRAPHEDGEARRAYPROPERTY_IMPL", getGraphEdgeArrayPropertyImpl.str());
    }
    // generate getAgentArrayVariable_ldg func implementation ($DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getAgentArrayVariableLDGImpl;
        if (!agent_variables.empty())
            getAgentArrayVariableLDGImpl <<             "    const size_t i = (index * N) + array_index;\n";
        for (std::pair<std::string, RTCVariableProperties> element : agent_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {  // GLM does not support __ldg() so should not use this
                getAgentArrayVariableLDGImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getAgentArrayVariableLDGImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getAgentArrayVariableLDGImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                getAgentArrayVariableLDGImpl << "                  DTHROW(\"Agent array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableLDGImpl << "                  return {};\n";
                getAgentArrayVariableLDGImpl << "              } else if (N != " << element.second.elements << ") {\n";
                getAgentArrayVariableLDGImpl << "                  DTHROW(\"Agent array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getAgentArrayVariableLDGImpl << "                  return {};\n";
                getAgentArrayVariableLDGImpl << "              } else if (array_index >= " << element.second.elements << ") {\n";
                getAgentArrayVariableLDGImpl << "                  DTHROW(\"Agent array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getAgentArrayVariableLDGImpl << "                  return {};\n";
                getAgentArrayVariableLDGImpl << "              }\n";
                getAgentArrayVariableLDGImpl << "#endif\n";
                getAgentArrayVariableLDGImpl << "                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << agent_data_offset + (ct * sizeof(void*)) << "))) + i);\n";
                getAgentArrayVariableLDGImpl << "           };\n";
                ++ct;  // Prev was part of the return line, but don't want confusion
            } else {
                ++ct;
            }
        }
        getAgentArrayVariableLDGImpl <<         "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getAgentArrayVariableLDGImpl <<         "           DTHROW(\"Agent array variable '%s' was not found during getVariable().\\n\", name);\n";
        getAgentArrayVariableLDGImpl <<         "#endif\n";
        getAgentArrayVariableLDGImpl <<         "           return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETAGENTARRAYVARIABLE_LDG_IMPL", getAgentArrayVariableLDGImpl.str());
    }
    // generate getMessageArrayVariable func implementation ($DYNAMIC_GETMESSAGEARRAYVARIABLE_LDG_IMPL)
    {
        size_t ct = 0;
        std::stringstream getMessageArrayVariableLDGImpl;
        if (!messageIn_variables.empty())
            getMessageArrayVariableLDGImpl << "    const size_t i = (index * N) + array_index;\n";
        for (const auto& element : messageIn_variables) {
            RTCVariableProperties props = element.second;
            if (props.read && props.elements > 1) {  // GLM does not support __ldg() so should not use this
                getMessageArrayVariableLDGImpl << "          if (strings_equal(name, \"" << element.first << "\")) {\n";
                getMessageArrayVariableLDGImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
                getMessageArrayVariableLDGImpl << "              if(sizeof(T) != " << element.second.type_size << ") {\n";
                getMessageArrayVariableLDGImpl << "                  DTHROW(\"Message array variable '%s' type mismatch during getVariable().\\n\", name);\n";
                getMessageArrayVariableLDGImpl << "                  return {};\n";
                getMessageArrayVariableLDGImpl << "              } else if (N != " << element.second.elements << ") {\n";
                getMessageArrayVariableLDGImpl << "                  DTHROW(\"Message array variable '%s' length mismatch during getVariable().\\n\", name);\n";
                getMessageArrayVariableLDGImpl << "                  return {};\n";
                getMessageArrayVariableLDGImpl << "              } else if (array_index >= " << element.second.elements << ") {\n";
                getMessageArrayVariableLDGImpl << "                  DTHROW(\"Message array variable '%s', index %d is out of bounds during getVariable().\\n\", name, array_index);\n";
                getMessageArrayVariableLDGImpl << "                  return {};\n";
                getMessageArrayVariableLDGImpl << "              }\n";
                getMessageArrayVariableLDGImpl << "#endif\n";
                getMessageArrayVariableLDGImpl << "                return (T) __ldg((*static_cast<T**>(static_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << messageIn_data_offset + (ct * sizeof(void*)) << "))) + i);\n";
                getMessageArrayVariableLDGImpl << "           };\n";
                ++ct;  // Prev was part of the return line, but don't want confusion
            } else {
                ++ct;
            }
        }
        getMessageArrayVariableLDGImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getMessageArrayVariableLDGImpl << "           DTHROW(\"Message array variable '%s' was not found during getVariable().\\n\", name);\n";
        getMessageArrayVariableLDGImpl << "#endif\n";
        getMessageArrayVariableLDGImpl << "           return {};\n";
        setHeaderPlaceholder("$DYNAMIC_GETMESSAGEARRAYVARIABLE_LDG_IMPL", getMessageArrayVariableLDGImpl.str());
    }

    // generate getGraphHash func implementation ($DYNAMIC_GETGRAPHHASH_IMPL)
    // This is bespoke to RTC Curve, as in place of getVariableIndex()
    {
        std::set<std::string> graphs_added;
        std::stringstream getGraphHashImpl;
        for (const auto& element : directedGraph_vertexProperties) {
            if (graphs_added.find(element.first.first) == graphs_added.end()) {
                getGraphHashImpl << "          if (strings_equal(graphName, \"" << element.first.first << "\")) {\n";
                getGraphHashImpl << "                return " << Curve::variableRuntimeHash(element.first.first) << ";\n";
                getGraphHashImpl << "          }\n";
                graphs_added.insert(element.first.first);
            }
        }
        getGraphHashImpl << "          return UNKNOWN_GRAPH;\n";
        setHeaderPlaceholder("$DYNAMIC_GETGRAPHHASH_IMPL", getGraphHashImpl.str());
    }

    // generate getVariableCount func implementation ($DYNAMIC_GETVARIABLECOUNT_IMPL)
    {
        std::stringstream getVariableCountImpl;
        for (const auto& element : directedGraph_vertexProperties) {
            getVariableCountImpl << "    if (namespace_hash == " << (Curve::variableRuntimeHash(element.first.first) ^ Curve::variableRuntimeHash("_environment_directed_graph_vertex")) << " && strings_equal(variableName, \"" << element.first.second << "\"))\n";
            getVariableCountImpl << "        return reinterpret_cast<unsigned int*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << count_data_offset << "))["<< element.second.count_index <<"];";
        }
        for (const auto& element : directedGraph_edgeProperties) {
            getVariableCountImpl << "    if (namespace_hash == " << (Curve::variableRuntimeHash(element.first.first) ^ Curve::variableRuntimeHash("_environment_directed_graph_edge")) << " && strings_equal(variableName, \"" << element.first.second << "\"))\n";
            getVariableCountImpl << "        return reinterpret_cast<unsigned int*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << count_data_offset << "))[" << element.second.count_index << "];";
        }
        // Note, the below are currently unused so untested (but support should be functional)
        for (const auto& element : agent_variables) {
            getVariableCountImpl << "    if (namespace_hash == 0 && strings_equal(variableName, \"" << element.first << "\"))\n";
            getVariableCountImpl << "        return reinterpret_cast<unsigned int*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << count_data_offset << "))[" << element.second.count_index << "];\n";
        }
        for (const auto& element : messageOut_variables) {
            getVariableCountImpl << "    if (namespace_hash == " << Curve::variableRuntimeHash("_message_out") << " && strings_equal(variableName, \"" << element.first << "\"))\n";
            getVariableCountImpl << "        return reinterpret_cast<unsigned int*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << count_data_offset << "))[" << element.second.count_index << "];\n";
        }
        for (const auto& element : messageIn_variables) {
            getVariableCountImpl << "    if (namespace_hash == " << Curve::variableRuntimeHash("_message_in") << " && strings_equal(variableName, \"" << element.first << "\"))\n";
            getVariableCountImpl << "        return reinterpret_cast<unsigned int*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << count_data_offset << "))[" << element.second.count_index << "];\n";
        }
        for (const auto& element : newAgent_variables) {
            getVariableCountImpl << "    if (namespace_hash == " << Curve::variableRuntimeHash("_agent_birth") << " && strings_equal(variableName, \"" << element.first << "\"))\n";
            getVariableCountImpl << "        return reinterpret_cast<unsigned int*>(reinterpret_cast<void*>(flamegpu::detail::curve::" << getVariableSymbolName() << " + " << count_data_offset << "))[" << element.second.count_index << "];\n";
        }
        getVariableCountImpl << "#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS\n";
        getVariableCountImpl << "    DTHROW(\"Curve variable with name '%s' was not found.\\n\", variableName);\n";
        getVariableCountImpl << "#endif\n";
        getVariableCountImpl << "          return 0;\n";
        setHeaderPlaceholder("$DYNAMIC_GETVARIABLECOUNT_IMPL", getVariableCountImpl.str());
    }
    setHeaderPlaceholder("$DYNAMIC_AGENT_NAME", this->agentName);
    setHeaderPlaceholder("$DYNAMIC_AGENT_STATE", this->agentState);
}
void CurveRTCHost::initDataBuffer() {
    if (data_buffer_size == 0 || h_data_buffer) {
        THROW exception::InvalidOperation("CurveRTCHost::initDataBuffer() should only be called once, during the init chain.\n");
    }
    // Alloc buffer
    gpuErrchk(cudaMallocHost(&h_data_buffer, data_buffer_size));
    // Notify all variables of their ptr to store data in cache
    size_t ct = 0;
    for (auto &element : agent_variables) {
        element.second.h_data_ptr = h_data_buffer + agent_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto &element : messageOut_variables) {
        element.second.h_data_ptr = h_data_buffer + messageOut_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto &element : messageIn_variables) {
        element.second.h_data_ptr = h_data_buffer + messageIn_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto &element : newAgent_variables) {
        element.second.h_data_ptr = h_data_buffer + newAgent_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto& element : directedGraph_vertexProperties) {
        element.second.h_data_ptr = h_data_buffer + directedGraphVertex_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto& element : directedGraph_edgeProperties) {
        element.second.h_data_ptr = h_data_buffer + directedGraphEdge_data_offset + (ct++ * sizeof(void*));
    }
    ct = 0;
    for (auto& element : RTCEnvMacroProperties) {
        element.second.h_data_ptr = h_data_buffer + envMacro_data_offset + (ct++ * sizeof(void*));
        // Env macro properties don't update, so fill them as we go
        memcpy(element.second.h_data_ptr, &element.second.d_ptr, sizeof(void*));
    }
}

void CurveRTCHost::setFileName(const std::string &filename) {
    setHeaderPlaceholder("$FILENAME", filename);
}

std::string CurveRTCHost::getDynamicHeader(const size_t env_buffer_len) {
    initHeaderEnvironment(env_buffer_len);
    initHeaderSetters();
    initHeaderGetters();
    initDataBuffer();
    return header;
}

void CurveRTCHost::setHeaderPlaceholder(std::string placeholder, std::string dst) {
    // replace placeholder with dynamically generated variables string
    size_t pos = header.find(placeholder);
    if (pos != std::string::npos) {
        header.replace(pos, placeholder.length(), dst);
    } else {
        THROW exception::UnknownInternalError("String (%s) not found when creating dynamic version of curve for RTC: in CurveRTCHost::setHeaderPlaceholder", placeholder.c_str());
    }
}

std::string CurveRTCHost::getVariableSymbolName() {
    std::stringstream name;
    name << "rtc_env_data_curve";
    return name.str();
}

std::string CurveRTCHost::demangle(const char* verbose_name) {
#ifndef _MSC_VER
    std::string s = jitify::reflection::detail::demangle_cuda_symbol(verbose_name);
#else
    // Jitify removed the required demangle function, this is a basic clone of what was being done in earlier version
    // It's possible jitify::reflection::detail::demangle_native_type() would work, however that requires type_info, not type_index
    size_t index = 0;
    std::string s = verbose_name;
    while (true) {
        /* Locate the substring to replace. */
        index = s.find("class", index);
        if (index == std::string::npos) break;

        /* Make the replacement. */
        s.replace(index, 5, "     ");

        /* Advance index forward so the next iteration doesn't pick it up as well. */
        index += 5;
    }
#endif
    // Lambda function for trimming whitesapce as jitify demangle does not remove this
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
        }));
#ifdef _MSC_VER
    // int64_t is the only known problematic type in windows as it has a typeid().name() of __int64.
    // This can be manually replaced
    std::string int64_type = "__int64";
    std::string int64_type_fixed = "long long int";
    size_t start_pos = s.find(int64_type);
    if (!(start_pos == std::string::npos))
        s.replace(start_pos, int64_type.length(), int64_type_fixed);
#endif

    // map known basic types in
    return s;
}

std::string CurveRTCHost::demangle(const std::type_index& type) {
    return demangle(type.name());
}
void CurveRTCHost::updateEnvCache(const void *env_ptr, const size_t bufferLen) {
    if (bufferLen <= agent_data_offset) {
        memcpy(h_data_buffer, env_ptr, bufferLen);
    } else {
        THROW exception::OutOfBoundsException("Provided bufferlen exceeds initialised env buffer len! %llu > %llu, "
        "in CurveRTCHost::updateEnvCache().",
            bufferLen, agent_data_offset);
    }
}
void CurveRTCHost::updateDevice_async(const jitify::experimental::KernelInstantiation& instance, cudaStream_t stream) {
    // Move count buffer into h_data_buffer first
    memcpy(h_data_buffer + count_data_offset, count_buffer.data(), count_buffer.size() * sizeof(unsigned int));
    // The namespace is required here, but not in other uses of getVariableSymbolName.
    std::string cache_var_name = std::string("flamegpu::detail::curve::") + getVariableSymbolName();
    CUdeviceptr d_var_ptr = instance.get_global_ptr(cache_var_name.c_str());
    gpuErrchkDriverAPI(cuMemcpyHtoDAsync(d_var_ptr, h_data_buffer, data_buffer_size, stream));
}

}  // namespace curve
}  // namespace detail
}  // namespace flamegpu
