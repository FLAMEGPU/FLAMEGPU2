#include "flamegpu/visualiser/color/HSVInterpolation.h"

#include <sstream>

#include "flamegpu/exception/FGPUException.h"

HSVInterpolation HSVInterpolation::REDGREEN(const std::string& variable_name, const float& _min_bound, const float& _max_bound) {
    auto rtn = HSVInterpolation(variable_name, 0.0f, 100.0f, 1.0f, 0.88f);
    rtn.setBounds(_min_bound, _max_bound);
    return rtn;
}
HSVInterpolation HSVInterpolation::GREENRED(const std::string& variable_name, const float& _min_bound, const float& _max_bound) {
    auto rtn = HSVInterpolation(variable_name, 100.0f, 0.0f, 1.0f, 0.88f);
    rtn.setBounds(_min_bound, _max_bound);
    return rtn;
}

HSVInterpolation::HSVInterpolation(const std::string &_variable_name, const float& hMin, const float& hMax, const float& s, const float& v)
    : hue_min(hMin)
    , hue_max(hMax)
    , saturation(s)
    , val(v)
    , variable_name(_variable_name) {
    if (hue_min < 0.0f || hue_min > 360.0f) {
        THROW InvalidArgument("%f is not a valid hue value, hue components must be in the inclusive range [0.0, 360.0], "
        "in HSVInterpolation::HSVInterpolation()\n", hue_min);
    }
    if (hue_max < 0.0f || hue_max > 360.0f) {
        THROW InvalidArgument("%f is not a valid hue value, hue components must be in the inclusive range [0.0, 360.0], "
        "in HSVInterpolation::HSVInterpolation()\n", hue_max);
    }
    if (hue_max == hue_min) {
        THROW InvalidArgument("hue_min and _hue_max must differ, "
        "in HSVInterpolation::HSVInterpolation()\n", hue_max);
    }
    if (saturation < 0.0f || saturation > 1.0f) {
        THROW InvalidArgument("%f is not a valid saturation, saturation must be in the inclusive range [0.0, 1.0], "
        "in HSVInterpolation::HSVInterpolation()\n", saturation);
    }
    if (val < 0.0f || val > 1.0f) {
        THROW InvalidArgument("%f is not a valid val, val must be in the inclusive range [0.0, 1.0], "
        "in HSVInterpolation::HSVInterpolation()\n", val);
    }
}
HSVInterpolation& HSVInterpolation::setBounds(const float& _min_bound, const float& _max_bound) {
    if (_min_bound >= _max_bound) {
        THROW InvalidArgument("max_bound (%f) must be greater than min_bound (%f), "
            "in HSVInterpolation::setBounds()\n",
            _max_bound, _min_bound);
    }
    min_bound = _min_bound;
    max_bound = _max_bound;
    return *this;
}
HSVInterpolation& HSVInterpolation::setWrapHue(const bool& _wrapHue) {
    wrap_hue = _wrapHue;
    return *this;
}
std::string HSVInterpolation::getSrc() const {
static const char* HEADER = R"###(
uniform samplerBuffer color_arg;
//hsv(0-360,0-1,0-1)
vec3 hsv2rgb(vec3 hsv) {
  if(hsv.g==0)//Grey
    return vec3(hsv.b);

  float h = hsv.r/60;
  int i = int(floor(h));
  float f = h-i;
  float p = hsv.b * (1-hsv.g);
  float q = hsv.b * (1-hsv.g * f);
  float t = hsv.b * (1-hsv.g * (1-f));
  switch(i) {
    case 0:
      return vec3(hsv.b,t,p);
    case 1:
      return vec3(q,hsv.b,p);
    case 2:
      return vec3(p,hsv.b,p);
    case 3:
      return vec3(p,q,hsv.b);
    case 4:
      return vec3(t,p,hsv.b);
    default: //case 5
      return vec3(hsv.b,p,q);
  }
}
)###";
    std::stringstream ss;
    ss << HEADER;
    ss << "vec4 calculateColor() {" << "\n";
    // Fetch the modifier from texture cache
    ss << "    float modifier = texelFetch(color_arg, gl_InstanceID).x;" << "\n";
    // Clamp the modifier to bounds
    ss << "    modifier = clamp(modifier, " << min_bound << ", " << max_bound << ");" << "\n";
    // Scale modifier to range [0.0, 1.0]
    ss << "    modifier = (modifier - " << min_bound << ") / " << (max_bound - min_bound) << ";" << "\n";
    // Apply HSV interpolation
    if (wrap_hue) {
        // Calculate hue
        if (hue_min > hue_max) {
            ss << "    modifier = " << hue_min << " + (modifier * " << (360.0f + hue_max - hue_min) << ");" << "\n";
        } else {
            ss << "    modifier = " << hue_min << " + (modifier * " << -(360.0f - (hue_max - hue_min)) << ");" << "\n";
        }
        // Wrap hue
        ss << "    modifier = modifier < 0.0 ? modifier + 360.0 : modifier;" << "\n";  // Only required by hue_min > hue_max?
        ss << "    modifier = modifier > 360.0 ? modifier - 360.0 : modifier;" << "\n";  // Only required by hue_max > hue_min?
        ss << "    return vec4(hsv2rgb(vec3(modifier, " << saturation << ", " << val << ")), 1.0);" << "\n";
    } else {
        if (hue_min < hue_max) {
            ss << "    return vec4(hsv2rgb(vec3(" << hue_min << " + (modifier * " << (hue_max - hue_min) << "), " << saturation << ", " << val << ")), 1.0);" << "\n";
        } else {
            ss << "    modifier = 1.0 - modifier;" << "\n";
            ss << "    return vec4(hsv2rgb(vec3(" << hue_max << " + (modifier * " << (hue_min - hue_max) << "), " << saturation << ", " << val << ")), 1.0);" << "\n";
        }
    }
    ss << "}" << "\n";
    return ss.str();
}
std::string HSVInterpolation::getSamplerName() const {
    return "color_arg";
}
std::string HSVInterpolation::getAgentVariableName() const {
    return variable_name;
}
std::type_index HSVInterpolation::getAgentVariableRequiredType() const {
    return std::type_index(typeid(float));
}
