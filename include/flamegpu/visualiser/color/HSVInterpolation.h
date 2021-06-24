#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_HSVINTERPOLATION_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_HSVINTERPOLATION_H_

#include <string>

#include "flamegpu/visualiser/color/ColorFunction.h"

namespace flamegpu {
namespace visualiser {

/**
 * Agent color function for mapping a floating point value to a HSV hue
 */
class HSVInterpolation : public ColorFunction {
 public:
    /**
     * 0 = Red, 1 = Green
     * @param variable_name agent variable of type float to map to the color
     * @param min_bound The value of the agent variable which should map to the Red
     * @param max_bound The value of the agent variable which should map to the Green
     */
    static HSVInterpolation REDGREEN(const std::string &variable_name, const float& min_bound = 0.0f, const float& max_bound = 1.0f);
    /**
     * 0 = Green, 1 = Red
     * @param variable_name agent variable of type float to map to the color
     * @param min_bound The value of the agent variable which should map to Green
     * @param max_bound The value of the agent variable which should map to Red
     */
    static HSVInterpolation GREENRED(const std::string& variable_name, const float& min_bound = 0.0f, const float& max_bound = 1.0f);
    /**
     * Constructs a HSV interpolation function generator
     * All components must be provided in the inclusive range [0.0, 1.0]
     * @param variable_name Name of the agent variable which maps to hue, the variable type must be float
     * @param hMin Hue value when the agent variable is 0.0
     * @param hMax Hue value when the agent variable is 1.0
     * @param s Saturation (the inverse amount of grey)
     * @param v Value (brightness)
     */
    HSVInterpolation(const std::string& variable_name, const float& hMin, const float& hMax, const float& s = 1.0f, const float& v = 0.88f);
    /**
     * Set the bounds to clamp an agent variable to before using for HSV interpolation
     * @param min_bound The agent variable value that should map to the minimum hue, must be smaller than max_bound
     * @param max_bound The agent variable value that should map to the maximum hue, must be larger than min_bound
     * @return Returns itself, so that you can chain the method (otherwise constructor would have too many optional args)
     * @throws exception::InvalidArgument if min_bound > max_bound
     * @note Defaults to (0.0, 1.0)
     */
    HSVInterpolation &setBounds(const float& min_bound, const float& max_bound);
    /**
     * If set to true, hue will interpolate over the 0/360 boundary
     * If set false, hue will interpolate without wrapping (e.g. if hMax is < hMin), hMin will be assigned to the lower_bound
     * By default this is set to false
     */
    HSVInterpolation& setWrapHue(const bool& _wrapHue);
    /**
     * Returns GLSL for a function that returns a color based on the configured HSV interpolation
     */
    std::string getSrc() const override;
    /**
     * Always returns "color_arg"
     */
    std::string getSamplerName() const override;
    /**
     * Returns variable_name
     */
    std::string getAgentVariableName() const override;
    /**
     * Returns std::type_index(typeid(float))
     */
    std::type_index getAgentVariableRequiredType() const override;

 private:
    /**
     * Bounds that the agent variable is clamped to
     * Defaults to [0.0, 1.0]
     */
    float min_bound = 0.0f, max_bound = 1.0f;
    /**
     * true, causes hue to interpolate over the 0/360 boundary
     * false, causes min_hue/max_hue to be swapped if max_hue < min_hue
     * Defaults to false
     */
    bool wrap_hue = false;
    /**
     * Hue must be in the inclusive range [0.0, 360.0] 
     */
    const float hue_min, hue_max;
    /**
     * Sat must be in the inclusive range [0.0, 1.0]
     */
    const float saturation;
    /**
     * Val must be in the inclusive range [0.0, 1.0]
     */
    const float val;
    /**
     * User specified name of the agent variable in the model which is used to interpolate the color
     */
    const std::string variable_name;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_HSVINTERPOLATION_H_
