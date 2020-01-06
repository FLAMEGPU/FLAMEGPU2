#include "flamegpu/runtime/utility/EnvironmentManager.cuh"

#include <cassert>
#include <map>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/utility/DeviceEnvironment.cuh"
#include "flamegpu/model/EnvironmentDescription.h"

/**
 * Internal namespace to hide __constant__ declarations from modeller
 */
namespace flamegpu_internal {
    /**
     * Managed by HostEnvironment, holds all environment properties
     */
    __constant__ char c_envPropBuffer[EnvironmentManager::MAX_BUFFER_SIZE];
    /**
     * Managed by HostEnvironment, returned whenever a failure state is reached
     * Think this exists because we cant return a reference to a constexpr
     */
    __constant__ uint64_t c_deviceEnvErrorPattern;
}  // namespace flamegpu_internal

const char EnvironmentManager::CURVE_NAMESPACE_STRING[23] = "ENVIRONMENT_PROPERTIES";

EnvironmentManager::EnvironmentManager() :
    curve(Curve::getInstance()),
    CURVE_NAMESPACE_HASH(curve.variableRuntimeHash(CURVE_NAMESPACE_STRING)),
    nextFree(0),
    m_freeSpace(EnvironmentManager::MAX_BUFFER_SIZE),
    freeFragments() {
    {
        void *t_c_buffer = nullptr;
        gpuErrchk(cudaGetSymbolAddress(&t_c_buffer, flamegpu_internal::c_envPropBuffer));
        c_buffer = reinterpret_cast<char*>(t_c_buffer);
        // printf("Env Prop Constant Cache Buffer: %p - %p\n", c_buffer, c_buffer + MAX_BUFFER_SIZE);
        assert(CURVE_NAMESPACE_HASH == DeviceEnvironment::CURVE_NAMESPACE_HASH());  // Host and Device namespace const's do not match
        // Setup device-side error pattern
        const uint64_t h_errorPattern = DeviceEnvironment::ERROR_PATTERN();
        gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::c_deviceEnvErrorPattern, reinterpret_cast<const void*>(&h_errorPattern), sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
    }
}


void EnvironmentManager::init(const std::string &model_name, const EnvironmentDescription &desc) {
    // Error if reinit
    for (auto &&i : properties) {
        if (i.first.first == model_name) {
            THROW EnvDescriptionAlreadyLoaded("Environment description with same model name '%s' is already loaded, "
                "in EnvironmentManager::init().",
                model_name.c_str());
        }
    }
    // Build a DefragMap to send to defragger method
    DefragMap orderedProperties;
    size_t newSize = 0;
    for (auto _i = desc.properties.begin(); _i != desc.properties.end(); ++_i) {
        const auto &i = _i->second;
        NamePair name = toName(model_name, _i->first);
        DefragProp prop = DefragProp(i.data.ptr, i.data.length, i.isConst, i.elements, i.type);
        const size_t typeSize = i.data.length / i.elements;
        orderedProperties.emplace(typeSize, std::make_pair(name, prop));
        newSize += i.data.length;
    }
    if (newSize > m_freeSpace) {
        // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
        // Arguably this check should be performed by init()
        THROW OutOfMemory("Insufficient EnvProperty memory to create new properties,"
            "in EnvironmentManager::init().");
    }
    // Defragment to rebuild it properly
    defragment(&orderedProperties);
}
void EnvironmentManager::free(const std::string &model_name) {
    curve.setNamespaceByHash(CURVE_NAMESPACE_HASH);
    for (auto &&i = properties.begin(); i != properties.end();) {
        if (i->first.first == model_name) {
            // Release from CURVE
            Curve::VariableHash cvh = toHash(i->first);
            curve.unregisterVariableByHash(cvh);
            // Drop from properties map
            i = properties.erase(i);
        } else {
            ++i;
        }
    }
    curve.setDefaultNamespace();
    // Defragment to clear up all the buffer items we didn't handle here
    defragment();
}

EnvironmentManager::NamePair EnvironmentManager::toName(const std::string &model_name, const std::string &var_name) {
    return std::make_pair(model_name, var_name);
}

/**
 * @note Not static, because eventually we might need to use curve singleton
 */
Curve::VariableHash EnvironmentManager::toHash(const NamePair &name) const {
    Curve::VariableHash model_cvh = curve.variableRuntimeHash(name.first.c_str());
    Curve::VariableHash var_cvh = curve.variableRuntimeHash(name.second.c_str());
    return model_cvh + var_cvh;
}

void EnvironmentManager::add(const NamePair &name, const char *ptr, const size_t &length, const bool &isConst, const size_type &elements, const std::type_index &type) {
    assert(elements > 0);
    const size_t typeSize = (length / elements);
    ptrdiff_t buffOffset = MAX_BUFFER_SIZE;
    // Allocate buffer space, using a free fragment
    for (auto it = freeFragments.begin(); it != freeFragments.end(); ++it) {
        const ptrdiff_t alignmentOffset = std::get<OFFSET>(*it) % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (std::get<LEN>(*it) + alignmentFix <= length) {
            // We can use this space!
            buffOffset = std::get<OFFSET>(*it) + alignmentFix;
            // Update freeFrags
            if (alignmentFix != 0) {
                freeFragments.push_back(OffsetLen(std::get<OFFSET>(*it), alignmentFix));
            }
            // Update nextFree
            if (std::get<LEN>(*it) == length) {
                // Remove
                freeFragments.erase(it);
            } else {
                // Shrink
                *it = { std::get<OFFSET>(*it) + length, std::get<LEN>(*it) - length };
            }
            break;
        }
    }
    // Allocate buffer space, using nextFree
    if (buffOffset == MAX_BUFFER_SIZE) {
        const ptrdiff_t alignmentOffset = nextFree % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (nextFree + alignmentFix + length < MAX_BUFFER_SIZE) {
            // Update freeFrags
            if (alignmentFix != 0) {
                freeFragments.push_back(OffsetLen(nextFree, alignmentFix));
            }
            // We can use this space!
            nextFree += alignmentFix;
            buffOffset = nextFree;
            nextFree += length;
        }
    }
    if (buffOffset == MAX_BUFFER_SIZE) {  // buffOffset hasn't changed from init value
        // defragment() and retry using nextFree
        defragment();
        const ptrdiff_t alignmentOffset = nextFree % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (nextFree + alignmentFix + length < MAX_BUFFER_SIZE) {
            // Update freeFrags
            if (alignmentFix != 0) {
                freeFragments.push_back(OffsetLen(nextFree, alignmentFix));
            }
            // We can use this space!
            nextFree += alignmentFix;
            buffOffset = nextFree;
            nextFree += length;
        } else {
            // Ran out of constant cache space!
            THROW OutOfMemory("Insufficient EnvProperty memory to create new property,"
                "in EnvironmentManager::add().");
        }
    }
    // Update FreeSpace
    m_freeSpace -= length;
    // Add to properties
    // printf("Constant '%s' created at offset: %llu, (%llu%%8), (%llu%%4)\n", name.c_str(), buffOffset, buffOffset % 8, buffOffset % 4);
    properties.emplace(name, EnvProp(buffOffset, length, isConst, elements, type));
    // Store data
    memcpy(hc_buffer + buffOffset, ptr, length);
    gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(const_cast<char*>(c_buffer + buffOffset)), reinterpret_cast<void*>(hc_buffer + buffOffset), length, cudaMemcpyHostToDevice));
    // Register in cuRVE
    curve.setNamespaceByHash(CURVE_NAMESPACE_HASH);
    Curve::VariableHash cvh = toHash(name);
    const auto CURVE_RESULT = curve.registerVariableByHash(cvh, reinterpret_cast<void*>(const_cast<char*>(c_buffer + buffOffset)), typeSize, elements);
    if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
        THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE"
            "in EnvironmentManager::add().");
    }
    curve.setDefaultNamespace();
}

void EnvironmentManager::defragment(DefragMap *mergeProperties) {
    // Build a multimap to sort the elements (to create natural alignment in compact form)
    DefragMap orderedProperties;
    for (auto &i : properties) {
        size_t typeLen = i.second.length / i.second.elements;
        orderedProperties.emplace(typeLen, std::make_pair(i.first, DefragProp(i.second)));
    }
    // Include any merge elements
    if (mergeProperties) {
        orderedProperties.insert(mergeProperties->cbegin(), mergeProperties->cend());
    }
    // Clear freefrags, so we can refill it with alignment junk
    freeFragments.clear();
    size_t spareFrags = 0;
    // Rebuild properties map into temporary buffer
    std::unordered_map<NamePair, EnvProp, NamePairHash> t_properties;
    char t_buffer[MAX_BUFFER_SIZE];
    ptrdiff_t buffOffset = 0;
    curve.setNamespaceByHash(CURVE_NAMESPACE_HASH);
    // Iterate largest vars first
    for (auto _i = orderedProperties.rbegin(); _i != orderedProperties.rend(); ++_i) {
        size_t typeSize = _i->first;
        auto &i = _i->second;
        // Handle alignment
        const ptrdiff_t alignmentOffset = buffOffset % typeSize;
        const ptrdiff_t alignmentFix = alignmentOffset != 0 ? typeSize - alignmentOffset : 0;
        if (alignmentOffset != 0) {
            freeFragments.push_back(OffsetLen(buffOffset, alignmentFix));
            buffOffset += alignmentFix;
            spareFrags += alignmentFix;
        }
        if (buffOffset + i.second.length <= MAX_BUFFER_SIZE) {
            // Setup constant in new position
            memcpy(t_buffer + buffOffset, i.second.data, i.second.length);
            t_properties.emplace(i.first, EnvProp(buffOffset, i.second.length, i.second.isConst, i.second.elements, i.second.type));
            // Update cuRVE (There isn't an update, so unregister and reregister)  // TODO: curveGetVariableHandle()?
            Curve::VariableHash cvh = toHash(i.first);
            //Only unregister variable if it's already registered
            if (!mergeProperties) {  // Merge properties are only provided on 1st init, when vars can't be unregistered
                curve.unregisterVariableByHash(cvh);
            } else {
                // Can this var be found inside mergeProps
                auto range = mergeProperties->equal_range(_i->first);
                bool isFound = false;
                for (auto w = range.first; w != range.second; ++w) {
                    if( w->second.first == _i->second.first) {
                        isFound = true;
                        break;
                    }
                }
                if(!isFound) {
                    curve.unregisterVariableByHash(cvh);
                }
            }
            const auto CURVE_RESULT = curve.registerVariableByHash(cvh, reinterpret_cast<void*>(const_cast<char*>(c_buffer + buffOffset)),
                typeSize, i.second.elements);
            if (CURVE_RESULT == Curve::UNKNOWN_VARIABLE) {
                THROW CurveException("curveRegisterVariableByHash() returned UNKNOWN_CURVE_VARIABLE, "
                    "in EnvironmentManager::add().");
            }
            // Increase buffer offset length that has been added
            buffOffset += i.second.length;
        } else {
            // Ran out of constant cache space! (this can only trigger when a DefragMap is passed)
            // Arguably this check should be performed by init()
            THROW OutOfMemory("Insufficient EnvProperty memory to create new properties, "
                "in EnvironmentManager::defragment(DefragMap).");
        }
    }
    curve.setDefaultNamespace();
    // Replace stored properties with temp
    std::swap(properties, t_properties);
    // Replace buffer on host
    memcpy(hc_buffer, t_buffer, buffOffset);
    // Replace buffer on device
    gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(const_cast<char*>(c_buffer)), reinterpret_cast<void*>(hc_buffer), buffOffset, cudaMemcpyHostToDevice));
    // Update m_freeSpace, nextFree
    nextFree = buffOffset;
    m_freeSpace = MAX_BUFFER_SIZE - buffOffset + spareFrags;
}
