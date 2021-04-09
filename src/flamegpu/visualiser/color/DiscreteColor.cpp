#include "flamegpu/visualiser/color/DiscreteColor.h"

#include "flamegpu/visualiser/color/Palette.h"

template<typename T>
DiscreteColor<T>::DiscreteColor(const std::string& _variable_name, const Color& _fallback)
    : std::map<T, Color>()
    , fallback(_fallback)
    , variable_name(_variable_name) { }
template<typename T>
DiscreteColor<T>::DiscreteColor(const std::string& _variable_name, const Palette& palette, const Color& _fallback, T offset, T stride)
    : DiscreteColor(_variable_name, _fallback) {
    // Construct map from palette
    for (const auto& i : palette) {
        this->emplace(offset, i);
        offset += stride;
    }
}
template<typename T>
DiscreteColor<T>::DiscreteColor(const std::string& _variable_name, const Palette& palette, T offset, T stride)
    : DiscreteColor(_variable_name, palette[palette.size()-1]) {
    // Construct map from palette
    for (size_t i = 0; i < palette.size() - 1; ++i) {
        this->emplace(offset, palette[i]);
        offset += stride;
    }
}

template<typename T>
std::string DiscreteColor<T>::getSamplerName() const {
    return "color_arg";
}
template<typename T>
std::string DiscreteColor<T>::getAgentVariableName() const {
    return variable_name;
}
template<typename T>
std::type_index DiscreteColor<T>::getAgentVariableRequiredType() const {
    return std::type_index(typeid(T));
}

template<typename T>
bool DiscreteColor<T>::validate() const {
    if (!fallback.validate())
        return false;
    for (const auto& m : *this)
        if (!m.second.validate())
            return false;
    return true;
}

// Force instantiate the 2 supported types
template class DiscreteColor<int32_t>;
template class DiscreteColor<uint32_t>;
