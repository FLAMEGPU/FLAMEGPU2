#ifndef INCLUDE_FLAMEGPU_EXCEPTION_FLAMEGPUEXCEPTION_H_
#define INCLUDE_FLAMEGPU_EXCEPTION_FLAMEGPUEXCEPTION_H_

#include <string>
#include <exception>
#include <cstdarg>
#include <cstdio>

namespace flamegpu {
namespace exception {

/**
 * If this macro is used instead of 'throw', FLAMEGPUException will 
 * prepend '__FILE__ (__LINE__): ' to err_message 
 */
#define THROW flamegpu::exception::FLAMEGPUException::setLocation(__FILE__, __LINE__); throw

/*! Class for unknown exceptions thrown*/
class UnknownError : public std::exception {};

/*! Base class for exceptions thrown */
class FLAMEGPUException : public std::exception {
 public:
    /**
     * A constructor
     * @brief Constructs the FLAMEGPUException object
     * @note Attempts to append '__FILE__ (__LINE__): ' to err_message
     */
     FLAMEGPUException();
    /**
     * @brief Returns the explanatory string
     * @return Pointer to a nullptr-terminated string with explanatory information. The pointer is guaranteed to be valid at least until the exception object from which it is obtained is destroyed, or until a non-const member function on the FLAMEGPUException object is called.
     */
     const char *what() const noexcept override;

     virtual const char* exception_type() const = 0;

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
 * Macro for generating common class body for derived classes of FLAMEGPUException
 * _DEBUG builds will print the error to stderr
 */
#ifdef _DEBUG
#define DERIVED_FLAMEGPUException(name, default_message)\
class name : public FLAMEGPUException {\
 public:\
    explicit name(const char *format = default_message, ...) {\
        va_list argp;\
        va_start(argp, format);\
        err_message += parseArgs(format, argp);\
        va_end(argp);\
        fprintf(stderr, "%s\n", err_message.c_str()); \
    }\
    const char* exception_type() const override {\
        return #name;\
    }\
}
#else
#define DERIVED_FLAMEGPUException(name, default_message)\
class name : public FLAMEGPUException {\
 public:\
    explicit name(const char *format = default_message, ...) {\
        va_list argp;\
        va_start(argp, format);\
        err_message += parseArgs(format, argp);\
        va_end(argp);\
    }\
    const char* exception_type() const override {\
        return #name;\
    }\
}
#endif



/////////////////////
// Derived Classes //
/////////////////////

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid input file.
 *  where the input file does not exist or cannot be read by the program.
 */
DERIVED_FLAMEGPUException(CUDAError, "CUDA returned an error code!");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to unsuitable variable names
 */
DERIVED_FLAMEGPUException(ReservedName, "Variable names cannot begin with the character '_'.");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid input file.
 *  where the input file does not exist or cannot be read by the program.
 */
DERIVED_FLAMEGPUException(InvalidInputFile, "Invalid Input File");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent variable type.
 * This could happen when retriving or setting a variable of differet type.
 */
DERIVED_FLAMEGPUException(InvalidVarType, "Bad variable type in agent instance set/get variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to unsupported variable types
 * This primarily occurs for agent array variables with host agent reductions
 */
DERIVED_FLAMEGPUException(UnsupportedVarType, "Variables of this type are not supported by function");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent state name.
 */
DERIVED_FLAMEGPUException(InvalidStateName, "Invalid agent state name");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to a weak ptr expiring unexpectedly
 */
DERIVED_FLAMEGPUException(InvalidParent, "Invalid parent");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent names
 */
DERIVED_FLAMEGPUException(InvalidAgentName, "Invalid agent name");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message names
 */
DERIVED_FLAMEGPUException(InvalidMessageName, "Invalid message name");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to message types misaligning
 */
DERIVED_FLAMEGPUException(InvalidMessageType, "Invalid message type");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent
 */
DERIVED_FLAMEGPUException(InvalidAgent, "Invalid agent");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message
 */
DERIVED_FLAMEGPUException(InvalidMessage, "Invalid message");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent memory variable type.
 */
DERIVED_FLAMEGPUException(InvalidAgentVar, "Invalid agent memory variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent state names.
 */
DERIVED_FLAMEGPUException(InvalidAgentState, "Invalid agent state");

/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to length mismatch of array variables.
*/
DERIVED_FLAMEGPUException(InvalidVarArrayLen, "Length of array variable does not match");
/**
* Defines a type of object to be thrown as exception.
* It reports errors that are due to accessing outside of the bounds of an array variable
*/
DERIVED_FLAMEGPUException(OutOfRangeVarArray, "Index is out of range of the array variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message memory variable type.
 */
DERIVED_FLAMEGPUException(InvalidMessageVar, "Invalid message memory variable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid message list.
 */
DERIVED_FLAMEGPUException(InvalidMessageData, "Invalid Message data");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid sub models
 */
DERIVED_FLAMEGPUException(InvalidSubModel, "Invalid SubModel");
/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to sub model name already being in use
 */
DERIVED_FLAMEGPUException(InvalidSubModelName, "Invalid SubModel Name, already in use");
/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to sub agent name not being recognised
 */
DERIVED_FLAMEGPUException(InvalidSubAgentName, "SubAgent name was not recognised");
/**
 * Defines a type of object to be thrown as exception.
 * It reports errors when a user adds an unsupported combination of items to a layer
 */
DERIVED_FLAMEGPUException(InvalidLayerMember, "Layer configuration unsupported");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent variable.
 */
DERIVED_FLAMEGPUException(InvalidCudaAgent, "CUDA agent not found. This should not happen");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA message variable.
 */
DERIVED_FLAMEGPUException(InvalidCudaMessage, "CUDA message not found. This should not happen");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent map size (i.e.map size is qual to zero).
 */
DERIVED_FLAMEGPUException(InvalidCudaAgentMapSize, "CUDA agent map size is zero");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent description.
 */
DERIVED_FLAMEGPUException(InvalidCudaAgentDesc, "CUDA Agent uses different agent description");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid CUDA agent state.
 */
DERIVED_FLAMEGPUException(InvalidCudaAgentState, "The state does not exist within the CUDA agent.");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid agent variable type. 
 * This could happen when retrieving or setting a variable of different type.
 */
DERIVED_FLAMEGPUException(InvalidAgentFunc, "Unknown agent function");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid function layer index.
 */
DERIVED_FLAMEGPUException(InvalidFuncLayerIndx, "Agent function layer index out of bounds!");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid memory capacity.
 */
DERIVED_FLAMEGPUException(InvalidMemoryCapacity, "Invalid Memory Capacity");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to invalid operation.
 */
DERIVED_FLAMEGPUException(InvalidOperation, "Invalid Operation");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to CUDA device.
 */
DERIVED_FLAMEGPUException(InvalidCUDAdevice, "Invalid CUDA Device");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due to CUDA device.
 */
DERIVED_FLAMEGPUException(InvalidCUDAComputeCapability, "Invalid CUDA Device Compute Capability");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due adding an init/step/exit function/condition to a simulation multiply
 */
DERIVED_FLAMEGPUException(InvalidHostFunc, "Invalid Host Function");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due unsuitable arguments
 */
DERIVED_FLAMEGPUException(InvalidArgument, "Invalid Argument Exception");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due environment property name already in use
 */
DERIVED_FLAMEGPUException(DuplicateEnvProperty, "Environment property of same name already exists");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that are due invalid environment property names
 */
DERIVED_FLAMEGPUException(InvalidEnvProperty, "Environment property of name does not exist");

/**
 * Defines a type of object to be thrown as exception.
 * It reports errors that an environment property has been accessed with the wrong type
 */
DERIVED_FLAMEGPUException(InvalidEnvPropertyType, "Environment property of name does not have same type");

/**
 * Defines a type of object to be thrown as exception.
 * It reports that a change to a constant environment property was attempted
 */
DERIVED_FLAMEGPUException(ReadOnlyEnvProperty, "Cannot modify environment properties marked as constant");

/**
* Defines a type of object to be thrown as exception.
* It reports that EnvironmentManager already holds data from a model description of the same name
*/
DERIVED_FLAMEGPUException(EnvDescriptionAlreadyLoaded, "Environment description with same model name already is already loaded.");

/**
 * Defines a type of object to be thrown as exception.
 * It reports that memory limits have been exceeded
 */
DERIVED_FLAMEGPUException(OutOfMemory, "Allocation failed, sufficient memory unavailable");

/**
 * Defines a type of object to be thrown as exception.
 * It reports that CURVE reported a failure
 */
DERIVED_FLAMEGPUException(CurveException, "Curve reported an error!");

/**
 * Defines an attempt to access outside the valid bounds of an array
 */
DERIVED_FLAMEGPUException(OutOfBoundsException, "Index exceeds bounds of array!");

/**
 * Defines an exception for errors reported by TinyXML
 */
DERIVED_FLAMEGPUException(TinyXMLError, "TinyXML returned an error code!");
/**
 * Defines an exception for errors reported by RapidJSON
 */
DERIVED_FLAMEGPUException(RapidJSONError, "RapidJSON returned an error code!");

/**
 * Defines an exception for errors when model components are mixed up
 */
DERIVED_FLAMEGPUException(DifferentModel, "Attempted to use member from a different model!");

/**
 * Defines an exception for errors when the provided file type is not supported
 */
DERIVED_FLAMEGPUException(UnsupportedFileType, "Cannot handle file type.");
/**
 * Defines an exception for internal errors which should only occur during development
 */
DERIVED_FLAMEGPUException(UnknownInternalError, "An unknown error occured within FLAME GPU lib.");

/**
 * Defines an exception for errors when two agents try to output an array message to the same index
 */
DERIVED_FLAMEGPUException(ArrayMessageWriteConflict, "Two messages attempted to write to the same index");
/**
 * Defines an exception for errors relted to visualisation
 */
DERIVED_FLAMEGPUException(VisualisationException, "An exception prevented the visualisation from working.");
/**
 * Defines when std::weak_ptr::lock() returns nullptr
 */
DERIVED_FLAMEGPUException(ExpiredWeakPtr, "Unable to convert weak pointer to shared pointer.");
/**
 * Defines an error reported from cuda device code (agent functions and agent function conditions)
 */
DERIVED_FLAMEGPUException(DeviceError, "Error reported from device code");
/**
 * Defines an error reported when versions do not match
 */
DERIVED_FLAMEGPUException(VersionMismatch, "Versions do not match");
/**
 * Defines an error reported when the expect input/output file path does not exist
 */
DERIVED_FLAMEGPUException(InvalidFilePath, "File does not exist.");
/**
 * Defines an exception indicating that the flamegpu::util::detail::Timer has been used incorrectly.
 */
DERIVED_FLAMEGPUException(TimerException, "Invalid use of Timer");
/**
 * Defines an error reported by AgentFunctionDependencyGraph if the graph is invalid
 */
DERIVED_FLAMEGPUException(InvalidDependencyGraph, "Agent function dependency graph is invalid");
/**
 * Defines an error when it is detected that multiple agents of the same type (even if in different states) share the same ID
 * This should not occur if the shared ID matches ID_NOT_SET
 */
DERIVED_FLAMEGPUException(AgentIDCollision, "Multiple agents of same type share an ID");
/**
 * Defines an error when runs fail during an ensemble's execution
 */
DERIVED_FLAMEGPUException(EnsembleError, "One of more runs failed during the ensemble's execution");

}  // namespace exception
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_EXCEPTION_FLAMEGPUEXCEPTION_H_
