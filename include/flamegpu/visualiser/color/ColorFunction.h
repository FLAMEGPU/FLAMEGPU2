#ifndef INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLORFUNCTION_H_
#define INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLORFUNCTION_H_

#include <string>
#include <typeindex>

namespace flamegpu {
namespace visualiser {

/**
 * Interface for generating shader code for a function that generates a color during model execution
 */
class ColorFunction {
 public:
    /**
     * Returns GLSL source code for a function with the prototype
     * vec4 calculateColor()
     * It may optionally also include a definition for a uniform samplerBuffer
     * If a uniform samplerBuffer is included, it's identifier should be returned by getSamplerName() 
     */
    virtual std::string getSrc() const = 0;
    /**
     * If the shader source contains a samplerBuffer definition (e.g. of a single float/int)
     * This should be the identifier so that the buffer can be bound to it
     * Otherwise empty string
     */
    virtual std::string getSamplerName() const { return ""; }
    /**
     * If the shader source contains a samplerBuffer definition
     * This should be the name of the agent variable so that the buffer can be bound to it
     * Otherwise empty string
     */
    virtual std::string getAgentVariableName() const { return ""; }
    /**
     * If the shader source contains a samplerBuffer definition
     * This should be the type of the agent variable so that the buffer can be bound to it
     * Otherwise void typeid is returned.
     */
    virtual std::type_index getAgentVariableRequiredType() const { return std::type_index(typeid(void)); }
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLORFUNCTION_H_
