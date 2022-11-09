#ifndef INCLUDE_FLAMEGPU_VISUALISER_PANELVIS_H_
#define INCLUDE_FLAMEGPU_VISUALISER_PANELVIS_H_
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <set>

#include "flamegpu/model/EnvironmentData.h"
#include "flamegpu/visualiser/config/ModelConfig.h"

namespace flamegpu {
namespace visualiser {

/**
 * This class serves as an interface for managing an instance of PanelConfig
 * It allows elements to be specified for a UI panel for the visualisation
 */
class PanelVis {
 public:
    /**
     * @param _m Reference which this interface manages
     * @param _environment Environment description to be used for validating correctness of requested environment UI elements
     * @note This should only be constructed by ModelVis
     * @see ModelVis::newUIPanel()
     */
    PanelVis(std::shared_ptr<PanelConfig> _m, std::shared_ptr<EnvironmentData> _environment);
    /**
     * Add a header to begin a collapsible group of elements
     *
     * @param header_text Text of the header
     * @param begin_open If true, the section will not begin in a collapsed state
     */
    void newSection(const std::string& header_text, bool begin_open = true);
    /**
     * End a section, the following elements are not part of the previous collapsible section
     * @note If this is not preceded by a call to newSection() it will have no effect
     */
    void newEndSection();
    /**
     * Add a label containing a fixed string
     *
     * @param label_text Text of the label
     */
    void newStaticLabel(const std::string& label_text);
    /**
     * Add a separator, creates a horizontal line between two consecutive elements
     * Useful is organising a single panel into multiple sections
     */
    void newSeparator();
    /**
     * Add a slider which displays the named environment property and allows it's value to be updated by
     * dragging a slider through the defined range
     *
     * @param property_name Name of the affected environment property
     * @param min Minimum value of the slider
     * @param max Maximum value of the slider
     * @tparam T Type of the named environment property
     */
    template<typename T>
    void newEnvironmentPropertySlider(const std::string &property_name, T min, T max);
    /**
     * @copydoc PanelVis::newEnvironmentPropertySlider()
     * @param index Index of the specified element within the environment property array
     * @tparam N Optional, length of the named environment property. 0 can be provided to ignore this check
     * @note Environment property arrays cannot be added as a whole, each element must be specified individually
     */
    template<typename T, flamegpu::size_type N = 0>
    void newEnvironmentPropertySlider(const std::string& property_name, flamegpu::size_type index, T min, T max);
    /**
     * Add a drag element which displays the named environment property (or environment property array element) and allows it's value to be updated by
     * clicking and dragging the mouse left/right. Double click can also be used to enter a new value via typing.
     *
     * @param property_name Name of the affected environment property
     * @param min Minimum value that can be set
     * @param max Maximum value that can be set
     * @param speed Amount the value changes per pixel dragged
     * @tparam T Type of the named environment property
     */
    template<typename T>
    void newEnvironmentPropertyDrag(const std::string& property_name, T min, T max, float speed);
    /**
     * @copydoc PanelVis::newEnvironmentPropertyDrag(const std::string &, T, T, float)
     * @param index Index of the specified element within the environment property array
     * @tparam N Optional, length of the named environment property. 0 can be provided to ignore this check
     * @note Environment property arrays cannot be added as a whole, each element must be specified individually
     */
    template<typename T, flamegpu::size_type N = 0>
    void newEnvironmentPropertyDrag(const std::string& property_name, flamegpu::size_type index, T min, T max, float speed);
    /**
     * Add a input box (with +/- step buttons) which displays the named environment property (or environment property array element) and allows it's value to be updated by
     * clicking and dragging the mouse left/right. Double click can also be used to enter a new value via typing.
     *
     * @param property_name Name of the affected environment property
     * @param step Change per button click
     * @param step_fast Change per tick when holding button (?)
     * @tparam T Type of the named environment property
     */
    template<typename T>
    void newEnvironmentPropertyInput(const std::string& property_name, T step, T step_fast);
    /**
     * @copydoc PanelVis::newEnvironmentPropertyInput(const std::string &, T, T)
     * @param index Index of the specified element within the environment property array
     * @tparam N Optional, length of the named environment property. 0 can be provided to ignore this check
     * @note Environment property arrays cannot be added as a whole, each element must be specified individually
     */
    template<typename T, flamegpu::size_type N = 0>
    void newEnvironmentPropertyInput(const std::string& property_name, flamegpu::size_type index, T step, T step_fast);
    /**
     * Add a checkbox element which displays the named environment property (or environment property array element) and allows it's value to be toggled
     * between 0 and 1 by clicking.
     *
     * @param property_name Name of the affected environment property
     * @tparam T Type of the named environment property
     * @note This element only supports integer type properties
     */
    template<typename T>
    void newEnvironmentPropertyToggle(const std::string& property_name);
    /**
     * @copydoc PanelVis::newEnvironmentPropertyToggle(const std::string &)
     * @param index Index of the specified element within the environment property array
     * @tparam N Optional, length of the named environment property. 0 can be provided to ignore this check
     * @note Environment property arrays cannot be added as a whole, each element must be specified individually
     */
    template<typename T, flamegpu::size_type N = 0>
    void newEnvironmentPropertyToggle(const std::string& property_name, flamegpu::size_type index);

 private:
    /**
     * The model description to validate requests against
     */
    const std::unordered_map<std::string, EnvironmentData::PropData> env_properties;
    /**
     * The panel data which this class acts as an interface for managing
     */
    std::shared_ptr<PanelConfig> m;
    /**
     * Each property can only be added to a specific panel once
     * ImGui appears to use their names to classify input, so dupes cause odd behaviour
     */
    std::set<std::pair<std::string, flamegpu::size_type>> added_properties;
};
template<typename T, flamegpu::size_type N>
void PanelVis::newEnvironmentPropertySlider(const std::string& property_name, flamegpu::size_type index, T min, T max) {
    {  // Validate name/type/length
        const auto it = env_properties.find(property_name);
        if (it == env_properties.end()) {
            THROW exception::InvalidEnvProperty("Environment property '%s' was not found, "
                "in PanelVis::newEnvironmentPropertySlider()\n",
                property_name.c_str());
        } else if (it->second.data.type != std::type_index(typeid(T))) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
                "in PanelVis::newEnvironmentPropertySlider()\n",
                property_name.c_str(), std::type_index(typeid(T)).name(), it->second.data.type.name());
        } else if (N != 0 && N != it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' length mismatch %u != %u, "
                "in PanelVis::newEnvironmentPropertySlider()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (index >= it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' index out of bounds %u >= %u, "
                "in PanelVis::newEnvironmentPropertySlider()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (max < min) {
            THROW exception::InvalidArgument("max < min, "
                "in PanelVis::newEnvironmentPropertySlider()\n",
                property_name.c_str());
        } else if (!added_properties.insert({property_name, index}).second) {
            THROW exception::InvalidEnvProperty("Environment property '%s' has already been added to the panel, "
                "each environment property (or property array element) can only be added to a panel once, "
                "in PanelVis::newEnvironmentPropertySlider()\n",
                property_name.c_str());
        }
    }
    std::unique_ptr<PanelElement> ptr = std::unique_ptr<PanelElement>(new EnvPropertySlider<T>(property_name, index, min, max));
    m->ui_elements.push_back(std::move(ptr));
}
template<typename T>
void PanelVis::newEnvironmentPropertySlider(const std::string& property_name, T min, T max) {
    newEnvironmentPropertySlider<T, 0>(property_name, 0, min, max);
}
template<typename T, flamegpu::size_type N>
void PanelVis::newEnvironmentPropertyDrag(const std::string& property_name, flamegpu::size_type index, T min, T max, float speed) {
    {  // Validate name/type/length
        const auto it = env_properties.find(property_name);
        if (it == env_properties.end()) {
            THROW exception::InvalidEnvProperty("Environment property '%s' was not found, "
                "in PanelVis::newEnvironmentPropertyDrag()\n",
                property_name.c_str());
        } else if (it->second.data.type != std::type_index(typeid(T))) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
                "in PanelVis::newEnvironmentPropertyDrag()\n",
                property_name.c_str(), std::type_index(typeid(T)).name(), it->second.data.type.name());
        } else if (N != 0 && N != it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' length mismatch %u != %u, "
                "in PanelVis::newEnvironmentPropertyDrag()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (index >= it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' index out of bounds %u >= %u, "
                "in PanelVis::newEnvironmentPropertyDrag()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (max < min) {
            THROW exception::InvalidArgument("max < min, "
                "in PanelVis::newEnvironmentPropertyDrag()\n",
                property_name.c_str());
        } else if (!added_properties.insert({property_name, index}).second) {
            THROW exception::InvalidEnvProperty("Environment property '%s' has already been added to the panel, "
                "each environment property (or property array element) can only be added to a panel once, "
                "in PanelVis::newEnvironmentPropertyDrag()\n",
                property_name.c_str());
        }
    }
    std::unique_ptr<PanelElement> ptr = std::make_unique<EnvPropertyDrag<T>>(property_name, index, min, max, speed);
    m->ui_elements.push_back(std::move(ptr));
}
template<typename T>
void PanelVis::newEnvironmentPropertyDrag(const std::string& property_name, T min, T max, float speed) {
    newEnvironmentPropertyDrag<T, 0>(property_name, 0, min, max, speed);
}
template<typename T, flamegpu::size_type N>
void PanelVis::newEnvironmentPropertyInput(const std::string& property_name, flamegpu::size_type index, T step, T step_fast) {
    {  // Validate name/type/length
        const auto it = env_properties.find(property_name);
        if (it == env_properties.end()) {
            THROW exception::InvalidEnvProperty("Environment property '%s' was not found, "
                "in PanelVis::newEnvironmentPropertyInput()\n",
                property_name.c_str());
        } else if (it->second.data.type != std::type_index(typeid(T))) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
                "in PanelVis::newEnvironmentPropertyInput()\n",
                property_name.c_str(), std::type_index(typeid(T)).name(), it->second.data.type.name());
        } else if (N != 0 && N != it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' length mismatch %u != %u, "
                "in PanelVis::newEnvironmentPropertyInput()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (index >= it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' index out of bounds %u >= %u, "
                "in PanelVis::newEnvironmentPropertyInput()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (!added_properties.insert({property_name, index}).second) {
            THROW exception::InvalidEnvProperty("Environment property '%s' has already been added to the panel, "
                "each environment property (or property array element) can only be added to a panel once, "
                "in PanelVis::newEnvironmentPropertyInput()\n",
                property_name.c_str());
        }
    }
    std::unique_ptr<PanelElement> ptr = std::make_unique<EnvPropertyInput<T>>(property_name, index, step, step_fast);
    m->ui_elements.push_back(std::move(ptr));
}
template<typename T>
void PanelVis::newEnvironmentPropertyInput(const std::string& property_name, T step, T step_fast) {
    newEnvironmentPropertyInput<T, 0>(property_name, 0, step, step_fast);
}
template<typename T, flamegpu::size_type N>
void PanelVis::newEnvironmentPropertyToggle(const std::string& property_name, flamegpu::size_type index) {
    static_assert(std::is_integral<T>::value, "PanelVis::newEnvironmentPropertyToggle() only supports integer type properties.");
    {  // Validate name/type/length
        const auto it = env_properties.find(property_name);
        if (it == env_properties.end()) {
            THROW exception::InvalidEnvProperty("Environment property '%s' was not found, "
                "in PanelVis::newEnvironmentToggle()\n",
                property_name.c_str());
        } else if (it->second.data.type != std::type_index(typeid(T))) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' type mismatch '%s' != '%s', "
                "in PanelVis::newEnvironmentToggle()\n",
                property_name.c_str(), std::type_index(typeid(T)).name(), it->second.data.type.name());
        } else if (N != 0 && N != it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' length mismatch %u != %u, "
                "in PanelVis::newEnvironmentToggle()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (index >= it->second.data.elements) {
            THROW exception::InvalidEnvPropertyType("Environment property '%s' index out of bounds %u >= %u, "
                "in PanelVis::newEnvironmentToggle()\n",
                property_name.c_str(), N, it->second.data.elements);
        } else if (!added_properties.insert({property_name, index}).second) {
            THROW exception::InvalidEnvProperty("Environment property '%s' has already been added to the panel, "
                "each environment property (or property array element) can only be added to a panel once, "
                "in PanelVis::newEnvironmentToggle()\n",
                property_name.c_str());
        }
    }
    std::unique_ptr<PanelElement> ptr = std::make_unique<EnvPropertyToggle<T>>(property_name, index);
    m->ui_elements.push_back(std::move(ptr));
}
template<typename T>
void PanelVis::newEnvironmentPropertyToggle(const std::string& property_name) {
    newEnvironmentPropertyToggle<T, 0>(property_name, 0);
}
}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_PANELVIS_H_
