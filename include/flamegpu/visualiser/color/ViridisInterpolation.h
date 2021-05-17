#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_VIRIDISINTERPOLATION_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_VIRIDISINTERPOLATION_H_

#include <string>
#include <array>

#include "flamegpu/visualiser/color/ColorFunction.h"
#include "flamegpu/visualiser/color/Color.h"

namespace flamegpu {
namespace visualiser {

/**
 * Agent color function for mapping a floating point value to the Viridis palette
 * Viridis is a color blind friendly palette
 * Originally from BIDS/MatPlotLib: https://github.com/BIDS/colormap
 */
class ViridisInterpolation : public ColorFunction {
    friend class Viridis;

 public:
    /**
     * Constructs a HSV Viridis palette function generator
     * @param variable_name Name of the agent variable which maps to the color, the variable type must be float
     * @param min_bound The agent variable value that should map to the lowest color (Dark blue)
     * @param max_bound The agent variable value that should map to the highest color (Light yellow)
     * @note max_bound may be lower than min_bound if the palette should be reversed
     */
    explicit ViridisInterpolation(const std::string& variable_name, const float& min_bound = 0.0f, const float& max_bound = 1.0f);
    /**
     * Set the bounds to clamp an agent variable to before using for HSV interpolation
     * @param min_bound The agent variable value that should map to the minimum hue, must be smaller than max_bound
     * @param max_bound The agent variable value that should map to the maximum hue, must be larger than min_bound
     * @return Returns itself, so that you can chain the method (otherwise constructor would have too many optional args)
     * @throws InvalidArgument if min_bound > max_bound
     * @note Defaults to (0.0, 1.0)
     * @note This is somewhat redundant, but exists for interface parity with HSVInterpolation
     */
    ViridisInterpolation& setBounds(const float& min_bound, const float& max_bound);
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
    /**
     * Returns the raw colors used to generate the palette
     */
    static const std::array<const Color, 256>& rawColors();

 private:
    /**
     * Bounds that the agent variable is clamped to
     * Defaults to [0.0, 1.0]
     */
    float min_bound = 0.0f, max_bound = 1.0f;
    /**
     * User specified name of the agent variable in the model which is used to interpolate the color
     */
    const std::string variable_name;
    /**
     * If a user attempts to set bounds backwards, they are flipped and this set true
     */
    bool invert_palette;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_VIRIDISINTERPOLATION_H_
