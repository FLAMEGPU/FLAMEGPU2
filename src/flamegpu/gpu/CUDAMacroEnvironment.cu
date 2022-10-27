#include "flamegpu/gpu/CUDAMacroEnvironment.h"

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/gpu/CUDASimulation.h"
#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"
#include "flamegpu/util/detail/cuda.cuh"

namespace flamegpu {

CUDAMacroEnvironment::CUDAMacroEnvironment(const EnvironmentDescription& description, const CUDASimulation& _cudaSimulation)
    : cudaSimulation(_cudaSimulation) {
    for (const auto &p : description.getMacroPropertiesMap()) {
        properties.emplace(p.first, MacroEnvProp(p.second.type, p.second.type_size, p.second.elements));
    }
}

void CUDAMacroEnvironment::init(cudaStream_t stream) {
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
            gpuErrchk(cudaMemsetAsync(prop.second.d_ptr, 0, buffer_size, stream));
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));
}

void CUDAMacroEnvironment::init(const SubEnvironmentData& mapping, const CUDAMacroEnvironment &master_macro_env, cudaStream_t stream) {
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
                    gpuErrchk(cudaMemsetAsync(prop.second.d_ptr, 0, buffer_size, stream));
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
    gpuErrchk(cudaStreamSynchronize(stream));
}
void CUDAMacroEnvironment::free() {
    for (auto& prop : properties) {
        if (prop.second.d_ptr) {
            if (!prop.second.is_sub) {
                gpuErrchk(flamegpu::util::detail::cuda::cudaFree(prop.second.d_ptr));
            }
            prop.second.d_ptr = nullptr;
        }
    }
}
void CUDAMacroEnvironment::registerCurveVariables(detail::curve::HostCurve& curve) const {
    for (const auto& p : properties) {
        const unsigned int total_elements = p.second.elements[0] * p.second.elements[1] * p.second.elements[2] * p.second.elements[3];
        curve.registerSetMacroEnvironmentProperty(p.first, p.second.type, p.second.type_size, total_elements, p.second.d_ptr);
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
