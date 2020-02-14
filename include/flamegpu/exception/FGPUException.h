#ifndef INCLUDE_FLAMEGPU_EXCEPTION_FGPUEXCEPTION_H_
#define INCLUDE_FLAMEGPU_EXCEPTION_FGPUEXCEPTION_H_

#include <string>
#include <exception>
#include <cstdarg>

/**
 * If this macro is used instead of 'throw', FGPUException will 
 * prepend '__FILE__ (__LINE__): ' to err_message 
 */
#define THROW FGPUException::setLocation(__FILE__, __LINE__); throw

/*! Class for unknown exceptions thrown*/
class UnknownError : public std::exception {};

/*! Base class for exceptions thrown */
class FGPUException : public std::exception {
 public:
    /**
     * A constructor
     * @brief Constructs the FGPUException object
     * @note Attempts to append '__FILE__ (__LINE__): ' to err_message
     */
     FGPUException();
    /**
     * @brief Returns the explanatory string
     * @return Pointer to a nullptr-terminated string with explanatory information. The pointer is guaranteed to be valid at least until the exception object from which it is obtained is destroyed, or until a non-const member function on the FGPUException object is called.
     */
     const char *what() const noexcept override;

    /**
     * Sets internal members file and line, which are used by constructor
     */
     static void setLocation(const char *_file, const unsigned int &_line);

 protected:
    /**
     * Parses va_list to a string using vsnprintf
     */
     static std::string parseArgs(const char * format, va_list argp);
    std::string err_message;

 private:
    static const char *file;
    static unsigned int line;
};

/**
 * Macro for generating common class body for derived classes of FGPUException
 */
#define DERIVED_FGPUException(name, default_msg)\
class name : public FGPUException {\
 public:\
    explicit name(const char *format = default_msg, ...) {\
        va_list argp;\
        va_start(argp, format);\
        err_message += parseArgs(format, argp);\
        va_end(argp);\
    }\
}

/////////////////////
// Derived Classes //
/////////////////////

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid input file.
 *  where the input file does not exist or cannot be read by the program.
 */
DERIVED_FGPUException(CUDAError, "CUDA returned an error code!");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid input file.
 *  where the input file does not exist or cannot be read by the program.
 */
DERIVED_FGPUException(InvalidInputFile, "Invalid Input File");

/**
 * Defines a type of object to be thrown as exception.
 * It is used to report errors when hash list is full.
 */
DERIVED_FGPUException(InvalidHashList, "Hash list full. This should never happen");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent variable type.
 * This could happen when retriving or setting a variable of differet type.
 */
DERIVED_FGPUException(InvalidVarType, "Bad variable type in agent instance set/get variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent state name.
 */
DERIVED_FGPUException(InvalidStateName, "Invalid agent state name");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid map entry.
 */
DERIVED_FGPUException(InvalidMapEntry, "Missing entry in type sizes map. Something went bad.");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to a weak ptr expiring unexpectedly
 */
DERIVED_FGPUException(InvalidParent, "Invalid parent");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent names
 */
DERIVED_FGPUException(InvalidAgentName, "Invalid agent name");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message names
 */
DERIVED_FGPUException(InvalidMessageName, "Invalid message name");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to message types misaligning
 */
DERIVED_FGPUException(InvalidMessageType, "Invalid message type");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent
 */
DERIVED_FGPUException(InvalidAgent, "Invalid agent");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message
 */
DERIVED_FGPUException(InvalidMessage, "Invalid message");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent memory variable type.
 */
DERIVED_FGPUException(InvalidAgentVar, "Invalid agent memory variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message memory variable type.
 */
DERIVED_FGPUException(InvalidMessageVar, "Invalid message memory variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message list.
 */
DERIVED_FGPUException(InvalidMessageData, "Invalid Message data");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message list size.
 */
DERIVED_FGPUException(InvalidMessageSize, "Invalid Message List size");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent variable.
 */
DERIVED_FGPUException(InvalidCudaAgent, "CUDA agent not found. This should not happen");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA message variable.
 */
DERIVED_FGPUException(InvalidCudaMessage, "CUDA message not found. This should not happen");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent map size (i.e.map size is qual to zero).
 */
DERIVED_FGPUException(InvalidCudaAgentMapSize, "CUDA agent map size is zero");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent description.
 */
DERIVED_FGPUException(InvalidCudaAgentDesc, "CUDA Agent uses different agent description");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent state.
 */
DERIVED_FGPUException(InvalidCudaAgentState, "The state does not exist within the CUDA agent.");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent variable type. 
 * This could happen when retrieving or setting a variable of different type.
 */
DERIVED_FGPUException(InvalidAgentFunc, "Unknown agent function");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid function layer index.
 */
DERIVED_FGPUException(InvalidFuncLayerIndx, "Agent function layer index out of bounds!");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid population data.
 */
DERIVED_FGPUException(InvalidPopulationData, "Invalid Population data");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid memory capacity.
 */
DERIVED_FGPUException(InvalidMemoryCapacity, "Invalid Memory Capacity");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid operation.
 */
DERIVED_FGPUException(InvalidOperation, "Invalid Operation");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to CUDA device.
 */
DERIVED_FGPUException(InvalidCUDAdevice, "Invalid CUDA Device");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due adding an init/step/exit function/condition to a simulation multiply
 */
DERIVED_FGPUException(InvalidHostFunc, "Invalid Host Function");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due unsuitable arguments
 */
DERIVED_FGPUException(InvalidArgument, "Invalid Argument Exception");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due environment property name already in use
 */
DERIVED_FGPUException(DuplicateEnvProperty, "Environment property of same name already exists");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due invalid environment property names
 */
DERIVED_FGPUException(InvalidEnvProperty, "Environment property of name does not exist");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that an environment property has been accessed with the wrong type
 */
DERIVED_FGPUException(InvalidEnvPropertyType, "Environment property of name does not have same type");

/**
 * Defines a type of object to be thrown as exception.
 * It reports that a change to a constant environment property was attempted
 */
DERIVED_FGPUException(ReadOnlyEnvProperty, "Cannot modify environment properties marked as constant");

/**
* Defines a type of object to be thrown as exception.
* It reports that EnvironmentManager already holds data from a model description of the same name
*/
DERIVED_FGPUException(EnvDescriptionAlreadyLoaded, "Environment description with same model name already is already loaded.");

/**
 * Defines a type of object to be thrown as exception.
 * It reports that memory limits have been exceeded
 */
DERIVED_FGPUException(OutOfMemory, "Allocation failed, sufficient memory unavailable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports that CURVE reported a failure
 */
DERIVED_FGPUException(CurveException, "Curve reported an error!");

/**
 * Defines an attempt to access outside the valid bounds of an array
 */
DERIVED_FGPUException(OutOfBoundsException, "Index exceeds bounds of array!");

/**
 * Defines an exception for errors reported by TinyXML
 */
DERIVED_FGPUException(TinyXMLError, "TinyXML returned an error code!");

/**
 * Defines an exception for errors when model components are mixed up
 */
DERIVED_FGPUException(DifferentModel, "Attempted to use member from a different model!");

/**
 * Defines an exception for errors when the provided file type is not supported
 */
DERIVED_FGPUException(UnsupportedFileType, "Cannot handle file type.");

#endif  // INCLUDE_FLAMEGPU_EXCEPTION_FGPUEXCEPTION_H_
