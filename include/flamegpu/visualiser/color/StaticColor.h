#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_STATICCOLOR_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_STATICCOLOR_H_

#include <string>

#include "flamegpu/visualiser/color/ColorFunction.h"
#include "flamegpu/visualiser/color/Color.h"

namespace flamegpu {
namespace visualiser {

/**
 * Creates a color function returning a static color
 * You probably don't need to use this class directly, instances of Color are implicitly converted to a StaticColor
 * @note Currently ignores alpha channel of colors as Alpha support isn't properly tested
 */
class StaticColor : public ColorFunction {
 public:
    /**
     * Constructs a static color function generator
     * All components must be provided in the inclusive range [0.0, 1.0]
     * @param rgba Color to represent
     */
    explicit StaticColor(const Color &rgba);
    /**
     * Returns a function returning a constant color in the form:
     * vec4 calculateColor() {
     *   return vec4(1.0, 0.0, 0.0, 1.0);
     * }
     */
    std::string getSrc(unsigned int) const override;

 private:
    /**
     * Not possible where variable isn't used, so hide
     */
    using ColorFunction::setAgentArrayVariableElement;
    /**
     * Shader controls RGBA values, but currently we only expose RGB (A support is somewhat untested)
     */
    Color rgba;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_STATICCOLOR_H_
