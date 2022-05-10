#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

#include <iostream>

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/util/nvtx.h"

namespace flamegpu {

void EnvironmentManager::init(const EnvironmentDescription& desc) {
    if (properties.size()) {
        THROW exception::EnvDescriptionAlreadyLoaded("Environment manager has already been initialised, in EnvironmentManager::init().");
    } else if (!desc.properties.size()) {
        return;
    }
    // Build a DefragMap to order the properties in reverse type size for efficient alignment packing
    DefragMap orderedProperties;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        const auto& i = _i->second;
        DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.data.elements, i.data.type);
        const size_t typeSize = i.data.length / i.data.elements;
        orderedProperties.emplace(std::make_pair(typeSize, _i->first), prop);
    }
    // Iterate the ordered properties to decide their offsets and calculate the required buffer length (with alignment)
    size_t newSize = 0;
    for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
        auto& i = _i->second;
        size_t alignmentSize = _i->first.first;  // aka typeSize
        // Handle alignment
        const ptrdiff_t alignmentOffset = newSize % alignmentSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? alignmentSize - alignmentOffset : 0;
        newSize += alignmentFix;
        i.offset = newSize;
        newSize += i.length;
    }
    h_buffer_len = newSize;
    h_buffer = static_cast<char*>(malloc(newSize));
    // Now copy the items into the newly allocated buffer and build the cache
    for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
        auto& i = _i->second;
        const std::string& name = _i->first.second;
        // Setup property in new position
        memcpy(h_buffer + i.offset, i.data, i.length);
        properties.emplace(name, EnvProp(i.offset, i.length, i.isConst, i.elements, i.type));
    }
    gpuErrchk(cudaMalloc(&d_buffer, newSize));
}
void EnvironmentManager::init(const EnvironmentDescription& desc, const std::shared_ptr<EnvironmentManager>& parent_environment, const SubEnvironmentData& mapping) {
    init(desc);
    // Iterate and link up all mapped properties with parent model's environment
    std::set<std::string> new_mapped_props;
    for (const auto &mapped_prop : mapping.properties) {
        // Notify parent to map this property
        parent_environment->linkMappedProperty(mapped_prop.second, mapped_prop.first, shared_from_this());
        mapped_parent_properties.emplace(mapped_prop.first, MappedProp{mapped_prop.second, parent_environment, desc.properties.at(mapped_prop.first).isConst });
    }
}
void EnvironmentManager::linkMappedProperty(const std::string& parent_name, const std::string& sub_name, const std::shared_ptr<EnvironmentManager>& sub_environment) {
    // Check the mapping hasn't already been linked
    {
        const auto range = mapped_child_properties.equal_range(parent_name);
        for (auto i = range.first; i != range.second; ++i) {
            if (auto e = i->second.remoteEnv.lock()) {
                if (e.get() == sub_environment.get()) {
                    THROW exception::InvalidOperation("Environment property '%s' already has mapping configured with the provided sub_environment, "
                    "in EnvironmentManager::linkMappedProperty.", parent_name.c_str());
                }
            } else {
                THROW exception::ExpiredWeakPtr("Environment property '%s' has a mapping with an expired weak ptr, this should not occur, "
                    "in EnvironmentManager::linkMappedProperty.", parent_name.c_str());
            }
        }
    }
    // Link the mapping
    mapped_child_properties.emplace(parent_name, MappedProp{ sub_name, sub_environment });
    // If we also have this property mapped with a parent, propagate the mapping so the parent can update it directly
    const auto mpp_it = mapped_parent_properties.find(parent_name);
    if (mpp_it != mapped_parent_properties.end()) {
        if (const auto parent_environment = mpp_it->second.remoteEnv.lock()) {
            // Notify parent to map this property
            parent_environment->linkMappedProperty(mpp_it->second.remoteName, sub_name, sub_environment);
        } else {
            THROW exception::ExpiredWeakPtr("Environment property '%s' has a mapping with an expired weak ptr, this should not occur, "
                "in EnvironmentManager::linkMappedProperty.", parent_name.c_str());
        }
    } else {
        // Otherwise, propagate our current value directly to the sub_environment, so it begins with the correct value
        const auto p_it = properties.find(parent_name);
        if (p_it != properties.end()) {
            sub_environment->setPropertyDirect(sub_name, h_buffer + p_it->second.offset);
        } else {
            THROW exception::InvalidEnvProperty("Environment property '%s' was not found, "
                "in EnvironmentManager::linkMappedProperty().", parent_name.c_str());
        }
    }
}
void EnvironmentManager::propagateMappedPropertyValue(const std::string& property_name, const char* const src_ptr) {
    // Propagate all the way up the parent chain, then propagate updates back down
    // The update will be propagated back to child, however this is not an issue
    const auto mp_it = mapped_parent_properties.find(property_name);
    if (mp_it != mapped_parent_properties.end() && !mp_it->second.isConst) {
        if (const auto parent_environment = mp_it->second.remoteEnv.lock()) {
            // Notify parent to map this property
            parent_environment->propagateMappedPropertyValue(mp_it->second.remoteName, src_ptr);
        } else {
            THROW exception::ExpiredWeakPtr("Environment property '%s' has a mapping with an expired weak ptr, this should not occur, "
                "in EnvironmentManager::propagateMappedPropertyValue.", property_name.c_str());
        }
    } else {
        // Propagate locally, and then to children
        setPropertyDirect(property_name, src_ptr);
        const auto range = mapped_child_properties.equal_range(property_name);
        for (auto i = range.first; i != range.second; ++i) {
            if (auto remote_env = i->second.remoteEnv.lock()) {
                remote_env->setPropertyDirect(i->second.remoteName, src_ptr);
            } else {
                THROW exception::ExpiredWeakPtr("Environment property '%s' has a mapping with an expired weak ptr, this should not occur, "
                    "in EnvironmentManager::propagateMappedPropertyValue.", property_name.c_str());
            }
        }
    }
}
void EnvironmentManager::setPropertyDirect(const std::string& property_name, const char * const src_ptr) {
    const auto it = properties.find(property_name);
    if (it != properties.end()) {
        if (h_buffer + it->second.offset != src_ptr)  // Skip self copies
            memcpy(h_buffer + it->second.offset, src_ptr, it->second.length);
        d_buffer_ready = false;
    } else {
        THROW exception::InvalidEnvProperty("Environment property '%s' was not found, "
            "in EnvironmentManager::setProperty().", property_name.c_str());
    }
}
util::Any EnvironmentManager::getPropertyAny(const std::string& property_name) const {
    const EnvProp &prop = findProperty<void>(property_name, false, 0);
    return util::Any(h_buffer + prop.offset, prop.length, prop.type, prop.elements);
}

EnvironmentManager::~EnvironmentManager() {
    properties.clear();
    if (h_buffer) {
        free(h_buffer);
        h_buffer = nullptr;
    }
    if (d_buffer) {
        gpuErrchk(cudaFree(d_buffer));
        d_buffer = nullptr;
    }
    h_buffer_len = 0;
}
void EnvironmentManager::resetModel(const EnvironmentDescription& desc) {
    for (const auto &dp : desc.properties) {
        if (mapped_parent_properties.find(dp.first) == mapped_parent_properties.end()) {
            // Only reset properties which are not inherited from parent
            const auto& p = properties.at(dp.first);
            char * const dest_ptr = h_buffer + p.offset;
            memcpy(dest_ptr, dp.second.data.ptr, p.length);
            //  Notify children
            propagateMappedPropertyValue(dp.first, dest_ptr);
        }
    }
}
void EnvironmentManager::updateDevice_async(const cudaStream_t stream) const {
    if (!d_buffer_ready) {
        gpuErrchk(cudaMemcpyAsync(d_buffer, h_buffer, h_buffer_len, cudaMemcpyHostToDevice, stream));
        d_buffer_ready = true;
    }
}
}  // namespace flamegpu
