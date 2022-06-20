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
     * @param variable_array_len Length of the variable array
     */
    virtual std::string getSrc(unsigned int variable_array_len) const = 0;
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
     * This should be the name of the index of the element within the agent variable (0 if it's not an array variable)
     */
    virtual unsigned int getAgentArrayVariableElement() const { return element; }
    /**
     * If the shader source contains a samplerBuffer definition
     * This should be the type of the agent variable so that the buffer can be bound to it
     * Otherwise void typeid is returned.
     */
    virtual std::type_index getAgentVariableRequiredType() const { return std::type_index(typeid(void)); }
    /**
     * Specify the array variable's element to use
     * @param _element Index of the element within the array variable
     * @note Calling this is not required if the agent variable is not an array variable
     */
    void setAgentArrayVariableElement(const unsigned int _element) { element = _element; }

 protected:
    /**
     * The element of the array variable to use (else 0 if not array variable)
     */
    unsigned int element = 0;
};

}  // namespace visualiser
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_VISUALISER_COLOR_COLORFUNCTION_H_
