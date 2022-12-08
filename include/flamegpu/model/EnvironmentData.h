#ifndef INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDATA_H_

#include <memory>
#include <array>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>

#include "flamegpu/detail/Any.h"
#include "flamegpu/model/ModelData.h"

namespace flamegpu {

struct EnvironmentData {
    friend class ModelDescription;
    /**
     * Used to access the protected copy constructor
     */
    friend class std::shared_ptr<ModelData> ModelData::clone() const;
    /**
     * Constructor has access to privately add reserved items
     * Might be a cleaner way to do this
     */
    // friend class CUDASimulation;

    // friend class detail::SimRunner;
    // friend unsigned int CUDAEnsemble::simulate(const RunPlanVector& plans);
    /**
     * Holds all of the properties required to add a value to EnvironmentManager
     */
    struct PropData {
        /**
         * @param _is_const Is the property constant
         * @param _data The data to initially fill the property with
         */
        PropData(bool _is_const, const detail::Any& _data)
            : isConst(_is_const)
            , data(_data) { }
        bool isConst;
        const detail::Any data;
        bool operator==(const PropData& rhs) const {
            if (this == &rhs)
                return true;
            if (this->isConst != rhs.isConst
                || this->data.elements != rhs.data.elements
                || this->data.length != rhs.data.length
                || this->data.type != rhs.data.type)
                return false;
            if (this->data.ptr == rhs.data.ptr)
                return true;
            for (size_t i = 0; i < this->data.length; ++i) {
                if (static_cast<const char*>(this->data.ptr)[i] != static_cast<const char*>(rhs.data.ptr)[i])
                    return false;
            }
            return true;
        }
        bool operator!=(const PropData& rhs) const {
            return !operator==(rhs);
        }
    };
    /**
     * Holds all of the properties required to add a value to EnvironmentManager
     */
    struct MacroPropData {
        /**
         * @param _type The type index of the base type (e.g. typeid(float))
         * @param _type_size The size of the base type (e.g. sizeof(float))
         * @param _elements Number of elements in each dimension
         */
        MacroPropData(const std::type_index& _type, const size_t _type_size, const std::array<unsigned int, 4>& _elements)
            : type(_type)
            , type_size(_type_size)
            , elements(_elements) { }
        std::type_index type;
        size_t type_size;
        std::array<unsigned int, 4> elements;
        bool operator==(const MacroPropData& rhs) const {
            if (this == &rhs)
                return true;
            if (this->type != rhs.type
                || this->type_size != rhs.type_size
                || this->elements[0] != rhs.elements[0]
                || this->elements[1] != rhs.elements[1]
                || this->elements[2] != rhs.elements[2]
                || this->elements[3] != rhs.elements[3])
                return false;
            for (size_t i = 0; i < this->elements.size(); ++i) {
                if (this->elements[i] != rhs.elements[i])
                    return false;
            }
            return true;
        }
        bool operator!=(const MacroPropData& rhs) const {
            return !operator==(rhs);
        }
    };
    /**
     * Parent model
     */
    std::weak_ptr<const ModelData> model;
    /**
     * Main storage of all properties
     */
    std::unordered_map<std::string, PropData> properties{};
    /**
     * Main storage of all macroproperties
     */
    std::unordered_map<std::string, MacroPropData> macro_properties{};
    /**
     * Equality operator, checks whether EnvironmentData hierarchies are functionally the same
     * @returns True when environments are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const EnvironmentData&rhs) const;
    /**
     * Equality operator, checks whether EnvironmentData hierarchies are functionally different
     * @returns True when environments are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const EnvironmentData&rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    EnvironmentData(const EnvironmentData&other) = delete;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     */
    EnvironmentData(std::shared_ptr<const ModelData> model, const EnvironmentData&other);
    /**
     * Normal constructor, only to be called by ModelDescription
     */
    explicit EnvironmentData(std::shared_ptr<const ModelData> model);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_ENVIRONMENTDATA_H_
