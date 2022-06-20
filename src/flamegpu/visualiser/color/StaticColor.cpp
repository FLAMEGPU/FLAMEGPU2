// @todo - ifdef VISUALISATION
#include "flamegpu/visualiser/color/StaticColor.h"

#include <sstream>

#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace visualiser {

StaticColor::StaticColor(const Color& _rgba)
    : rgba(_rgba) {
    if (!rgba.validate()) {
        THROW exception::InvalidArgument("Provided color has invalid components, "
            "in StaticColor::StaticColor\n");
    }
}
std::string StaticColor::getSrc(unsigned int) const {
    std::stringstream ss;
    ss << "vec4 calculateColor() {" << "\n";
    ss << "    return vec4(" << rgba[0] << ", " << rgba[1] << ", " << rgba[2] << ", " << 1.0f << ");" << "\n";
    ss << "}" << "\n";
    return ss.str();
}

Color::operator StaticColor() const {
    return StaticColor{*this};
}

}  // namespace visualiser
}  // namespace flamegpu
