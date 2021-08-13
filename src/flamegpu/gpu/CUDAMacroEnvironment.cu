#include "flamegpu/gpu/CUDAMacroEnvironment.h"

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"

namespace flamegpu {

const char CUDAMacroEnvironment::MACRO_NAMESPACE_STRING[18] = "MACRO_ENVIRONMENT";

CUDAMacroEnvironment::CUDAMacroEnvironment(const EnvironmentDescription& description, const CUDASimulation& _cudaSimulation)
    : MACRO_NAMESPACE_HASH(detail::curve::Curve::variableRuntimeHash(MACRO_NAMESPACE_STRING))
    , cudaSimulation(_cudaSimulation) {
    assert(MACRO_NAMESPACE_HASH == DeviceEnvironment::MACRO_NAMESPACE_HASH());  // Host and Device namespace const's do not match
    for (const auto &p : description.getMacroPropertiesMap()) {
        properties.emplace(p.first, MacroEnvProp(p.second.type, p.second.type_size, p.second.elements));
    }
}

void CUDAMacroEnvironment::init() {
    for (auto &prop : properties) {
        if (!prop.second.d_ptr) {
            size_t buffer_size = prop.second.type_size
                                     * prop.second.elements[0]
                                     * prop.second.elements[1]
                                     * prop.second.elements[2]
                                     * prop.second.elements[3];
#if !defined(SEATBELTS) || SEATBELTS
            buffer_size += sizeof(unsigned int);  // Extra uint is used as read-write flag by seatbelts
#endif
            gpuErrchk(cudaMalloc(&prop.second.d_ptr, buffer_size));
            gpuErrchk(cudaMemset(prop.second.d_ptr, 0, buffer_size));
        }
    }
    mapRuntimeVariables();
}

void CUDAMacroEnvironment::init(const SubEnvironmentData& mapping, const CUDAMacroEnvironment &master_macro_env) {
    // Map local properties
    for (auto& prop : properties) {
        if (!prop.second.d_ptr) {
            auto sub = mapping.macro_properties.find(prop.first);
            if (sub == mapping.macro_properties.end()) {
                // If it's a local macro property
                    size_t buffer_size = prop.second.type_size
                        * prop.second.elements[0]
                        * prop.second.elements[1]
                        * prop.second.elements[2]
                        * prop.second.elements[3];
#if !defined(SEATBELTS) || SEATBELTS
                    buffer_size += sizeof(unsigned int);  // Extra uint is used as read-write flag by seatbelts
#endif
                    gpuErrchk(cudaMalloc(&prop.second.d_ptr, buffer_size));
                    gpuErrchk(cudaMemset(prop.second.d_ptr, 0, buffer_size));
            } else {
                // If it's a mapped sub macro property
                auto mmp = master_macro_env.properties.find(sub->second);
                if (mmp != master_macro_env.properties.end()
                    && mmp->second.d_ptr
                    && mmp->second.elements == prop.second.elements
                    && mmp->second.type == prop.second.type) {
                    prop.second.d_ptr = mmp->second.d_ptr;
                    prop.second.is_sub = true;
                } else {
                    THROW exception::UnknownInternalError("Unable to initialise mapped macro property '%s' to '%s', this should not have failed, "
                    "in CUDAMacroEnvironment::init()\n",
                    prop.first.c_str(), sub->second.c_str());
                }
            }
        }
    }
    // Pass them all to CURVE
    mapRuntimeVariables();
}
void CUDAMacroEnvironment::free() {
    unmapRuntimeVariables();
    for (auto& prop : properties) {
        if (prop.second.d_ptr) {
            if (!prop.second.is_sub) {
                gpuErrchk(cudaFree(prop.second.d_ptr));
            }
            prop.second.d_ptr = nullptr;
        }
    }
}
void CUDAMacroEnvironment::purge() {
    for (auto& prop : properties)
        prop.second.d_ptr = nullptr;
}

void CUDAMacroEnvironment::mapRuntimeVariables() const {
    auto& curve = detail::curve::Curve::getInstance();
    // loop through the agents variables to map each variable name using cuRVE
    for (const auto& mmp : properties) {
        // map using curve
        const detail::curve::Curve::VariableHash var_hash = detail::curve::Curve::variableRuntimeHash(mmp.first.c_str());

        // get the agent variable size
        const unsigned int length = mmp.second.elements[0] * mmp.second.elements[1] * mmp.second.elements[2] * mmp.second.elements[3];

#ifdef _DEBUG
            const detail::curve::Curve::Variable cv = curve.registerVariableByHash(var_hash + MACRO_NAMESPACE_HASH + cudaSimulation.getInstanceID(), mmp.second.d_ptr, mmp.second.type_size, length);
            if (cv != static_cast<int>((var_hash + MACRO_NAMESPACE_HASH + cudaSimulation.getInstanceID()) % detail::curve::Curve::MAX_VARIABLES)) {
                fprintf(stderr, "detail::curve::Curve Warning: Environment macro property '%s' has a collision and may work improperly.\n", mmp.first.c_str());
            }
#else
            curve.registerVariableByHash(var_hash + MACRO_NAMESPACE_HASH + cudaSimulation.getInstanceID(), mmp.second.d_ptr, mmp.second.type_size, length);
#endif
    }
}

void CUDAMacroEnvironment::unmapRuntimeVariables() const {
    // loop through the agents variables to unmap each property using cuRVE
    for (const auto& mmp : properties) {
        const detail::curve::Curve::VariableHash var_hash = detail::curve::Curve::variableRuntimeHash(mmp.first.c_str());
        detail::curve::Curve::getInstance().unregisterVariableByHash(var_hash + MACRO_NAMESPACE_HASH + cudaSimulation.getInstanceID());
    }
}
void CUDAMacroEnvironment::mapRTCVariables(detail::curve::CurveRTCHost& curve_header) const {
    for (const auto &p : properties) {
        curve_header.registerEnvMacroProperty(p.first.c_str(), p.second.d_ptr, p.second.type.name(), p.second.type_size, p.second.elements);
    }
}
void CUDAMacroEnvironment::unmapRTCVariables(detail::curve::CurveRTCHost& curve_header) const {
    for (const auto &p : properties) {
        curve_header.unregisterEnvMacroProperty(p.first.c_str());
    }
}
#if !defined(SEATBELTS) || SEATBELTS
void CUDAMacroEnvironment::resetFlagsAsync(const std::vector<cudaStream_t> &streams) {
    unsigned int i = 0;
    for (const auto& prop : properties) {
        if (prop.second.d_ptr) {
            const size_t buffer_size = prop.second.type_size
                * prop.second.elements[0]
                * prop.second.elements[1]
                * prop.second.elements[2]
                * prop.second.elements[3];
            gpuErrchk(cudaMemsetAsync(static_cast<char*>(prop.second.d_ptr) + buffer_size, 0 , sizeof(unsigned int), streams[i++%streams.size()]));
        }
    }
    // Disable the sync here, users must sync themselves
    // if (properties.size()) {
    //     gpuErrchk(cudaDeviceSynchronize());
    // }
}
bool CUDAMacroEnvironment::getDeviceReadFlag(const std::string& property_name) {
    const auto prop = properties.find(property_name);
    if (prop == properties.end()) {
        THROW flamegpu::exception::InvalidEnvProperty("The environment macro property '%s' was not found, "
            "in CUDAMacroEnvironment::getDeviceReadFlag()\n",
            property_name.c_str());
    }
    const size_t buffer_size = prop->second.type_size
        * prop->second.elements[0]
        * prop->second.elements[1]
        * prop->second.elements[2]
        * prop->second.elements[3];
    unsigned int ret = 0;
    gpuErrchk(cudaMemcpy(&ret, static_cast<char*>(prop->second.d_ptr) + buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return (ret & 1u << 0);
}
bool CUDAMacroEnvironment::getDeviceWriteFlag(const std::string& property_name) {
    const auto prop = properties.find(property_name);
    if (prop == properties.end()) {
        THROW flamegpu::exception::InvalidEnvProperty("The environment macro property '%s' was not found, "
            "in CUDAMacroEnvironment::getDeviceReadFlag()\n",
            property_name.c_str());
    }
    const size_t buffer_size = prop->second.type_size
        * prop->second.elements[0]
        * prop->second.elements[1]
        * prop->second.elements[2]
        * prop->second.elements[3];
    unsigned int ret = 0;
    gpuErrchk(cudaMemcpy(&ret, static_cast<char*>(prop->second.d_ptr) + buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return (ret & 1u << 1);
}
unsigned int CUDAMacroEnvironment::getDeviceRWFlags(const std::string& property_name) {
    const auto prop = properties.find(property_name);
    if (prop == properties.end()) {
        THROW flamegpu::exception::InvalidEnvProperty("The environment macro property '%s' was not found, "
            "in CUDAMacroEnvironment::getDeviceReadFlag()\n",
            property_name.c_str());
    }
    const size_t buffer_size = prop->second.type_size
        * prop->second.elements[0]
        * prop->second.elements[1]
        * prop->second.elements[2]
        * prop->second.elements[3];
    unsigned int ret = 0;
    gpuErrchk(cudaMemcpy(&ret, static_cast<char*>(prop->second.d_ptr) + buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return ret;
}
#endif
}  // namespace flamegpu
