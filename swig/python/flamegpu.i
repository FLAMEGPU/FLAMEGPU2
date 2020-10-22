%begin %{
// define SWIG_PYTHON_INTERPRETER_NO_DEBUG on windows debug builds as pythonXX_d is not packaged unless built from source
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

// supress known warnings
//#pragma SWIG nowarn=325,302,401
//#pragma SWIG nowarn=302


// string support
%include <stl.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_unordered_map.i>
%include <std_array.i>

// typemaps for integer types (allows mapping of python types to stdint types)
%include <stdint.i>


// Directors required for callback functions
%module(directors="1") pyflamegpu


/* Instantiate the vector types used in the templated variable functions. Types must match those in TEMPLATE_VARIABLE_INSTANTIATE and TEMPLATE_VARIABLE_INSTANTIATE_N macros*/
%template(Int8Vector) std::vector<int8_t>;
%template(Int16Vector) std::vector<int16_t>;
%template(Int32Vector) std::vector<int32_t>;
%template(Int64Vector) std::vector<int64_t>;
%template(UInt8Vector) std::vector<uint8_t>;
%template(UInt16Vector) std::vector<uint16_t>;
%template(UInt32Vector) std::vector<uint32_t>;
%template(UInt64Vector) std::vector<uint64_t>;
%template(FloatVector) std::vector<float>;
%template(DoubleVector) std::vector<double>;
//%template(BoolVector) std::vector<bool>;

/**
 * TEMPLATE_VARIABLE_INSTANTIATE_FLOATS macro
 * Expands for floating point types
 */
%define TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(function, classfunction) 
// float and double
%template(function ## Float) classfunction<float>;
%template(function ## Double) classfunction<double>;
%enddef

/**
 * TEMPLATE_VARIABLE_INSTANTIATE macro
 * Expands for int types
 */
%define TEMPLATE_VARIABLE_INSTANTIATE_INTS(function, classfunction) 
// signed ints
%template(function ## Int16) classfunction<int16_t>;
%template(function ## Int32) classfunction<int32_t>;
%template(function ## Int64) classfunction<int64_t>;
// unsigned ints
%template(function ## UInt16) classfunction<uint16_t>;
%template(function ## UInt32) classfunction<uint32_t>;
%template(function ## UInt64) classfunction<uint64_t>;
// default int types
%template(function ## Int) classfunction<int>;
%template(function ## UInt) classfunction<unsigned int>;
%enddef

/**
 * TEMPLATE_VARIABLE_INSTANTIATE macro
 * Given a function name and a class::function specifier, this macro instaciates a typed version of the function for a set of basic types. 
 * E.g. TEMPLATE_VARIABLE_INSTANTIATE(function, SomeClass:function) will generate swig typed versions of the function like the following
 *    typedef SomeClass:function<int> functionInt;
 *    typedef SomeClass:function<float> functionFloat;
 *    ...
 */
%define TEMPLATE_VARIABLE_INSTANTIATE(function, classfunction) 
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(function, classfunction)
TEMPLATE_VARIABLE_INSTANTIATE_INTS(function, classfunction)
// char types
%template(function ## Int8) classfunction<int8_t>;
%template(function ## UInt8) classfunction<uint8_t>;
%template(function ## Char) classfunction<char>;
%template(function ## UChar) classfunction<unsigned char>;
// bool type (not supported causes error)
//%template(function ## Bool) classfunction<bool>;
%enddef


/**
 * TEMPLATE_SUM_INSTANTIATE macro
 * Specific template expansion for sum which allows different return types to avoid range issues 
 * Return with same type is not Instantiated. I.e. All int type use 64 bit signed returned types
 */
%define TEMPLATE_SUM_INSTANTIATE(sum_class) 
// float and double
%template(sumFloat) sum_class ## ::sum<float>;
%template(sumDouble) sum_class ## ::sum<double>;
// int types
%template(sumInt8) sum_class ## ::sumOutT<int8_t, int64_t>;
%template(sumUInt8) sum_class ## ::sumOutT<uint8_t, uint64_t>;
%template(sumInt16) sum_class ## ::sumOutT<int16_t, int64_t>;
%template(sumUInt16) sum_class ## ::sumOutT<uint16_t, uint64_t>;
%template(sumInt32) sum_class ## ::sumOutT<int32_t, int32_t>;
%template(sumUInt32) sum_class ## ::sumOutT<uint32_t, uint32_t>;
%template(sumInt64) sum_class ## ::sumOutT<int64_t, int64_t>;
%template(sumUInt64) sum_class ## ::sumOutT<uint64_t, uint64_t>;
// generic int types
%template(sumInt) sum_class ## ::sumOutT<int, int64_t>;
%template(sumUInt) sum_class ## ::sumOutT<unsigned int, uint64_t>;
// no chars or bool
%enddef

/* Compilation header includes */
%{
/* Includes the header in the wrapper code */
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubAgentDescription.h"
#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/gpu/CUDASimulation.h"

#include "flamegpu/exception/FGPUException.h"

//#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/runtime/flamegpu_host_agent_api.h"
#include "flamegpu/runtime/messaging.h"

#include "flamegpu/runtime/AgentFunction_shim.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
#include "flamegpu/runtime/HostFunctionCallback.h"
%}


/* Instantiate array types, these are required by message types when setting dimensions. */
%template(UIntArray2) std::array<unsigned int, 2>;
%template(UIntArray3) std::array<unsigned int, 3>;

/** Exception handling
 * FGPURuntimeException class is a wrapper class to replace specific instances of FGPUException. It is constructed with a error mesage and type which can be queried to give the original error and the original exception class.
 * The FGPURuntimeException is constructed form the original FGPUException in the handler and then translated into the python version.
 * The approach avoids having to wrap every exception class which extends FGPUException
 */
%exceptionclass FGPURuntimeException;
// It was hoped that the following would provide a nice repr implementation of the Python FGPURuntimeException objects. It does not.
//%feature("python:slot", "tp_str", functype="reprfunc") FGPURuntimeException::what
%inline %{

class FGPURuntimeException : public std::exception {
 public:
     FGPURuntimeException(std::string msg, std::string type) {
         err_message = msg;
         type_str = type;
     }
     const char* what() const noexcept {
         return err_message.c_str();
     }
     const char* type() const {
         return type_str.c_str();
     }
 protected:
    std::string err_message;
    std::string type_str;
};
%}
// Exception handling
%include "exception.i"

// swig director exceptions (handle python callback exceptions as C++ exceptions not Runtime errors)
%feature("director:except") {
  if ($error != NULL) {
    throw Swig::DirectorMethodException();
  }
}

%exception {
    try {
        $action
    }
    catch (FGPUException& e) {
        FGPURuntimeException *except = new FGPURuntimeException(std::string(e.what()), std::string(e.exception_type()));
        PyObject *err = SWIG_NewPointerObj(except, SWIGTYPE_p_FGPURuntimeException, 1);
        SWIG_Python_Raise(err, except.type(), SWIGTYPE_p_FGPURuntimeException); 
        SWIG_fail;
    }
    catch (Swig::DirectorException&) { 
        SWIG_fail; 
    }
    catch(const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, const_cast<char*>(e.what()) );
    }
    catch (...) {
        SWIG_exception(SWIG_RuntimeError, "Unknown Exception");
    } 
}

/* Create some type objects to obtain sizes and type info of flamegpu2 basic types.
 * It is not required to demangle type names. These can be used to compare directly with the type_index 
 * returned by the flamegpu2 library
 */
%nodefaultctor std::type_index;
%inline %{
    namespace Types{
        template <typename T>
        struct type_info{
	
	        static unsigned int size(){
		        return sizeof(T);
	        }
	
	        static const char* typeName() {
                return std::type_index(typeid(T)).name();
	        }

            static T empty() {
                return T();
            }

            static T fromValue(T value) {
                return value;
            }
        };

        const char* typeName(const std::type_index& type) {
            return type.name();
        }
    }
%}
%template(IntType) Types::type_info<int>;
%template(Int8Type) Types::type_info<int8_t>;
%template(Int16Type) Types::type_info<int16_t>;
%template(Int32Type) Types::type_info<int32_t>;
%template(Int64Type) Types::type_info<int64_t>;
%template(UIntType) Types::type_info<unsigned int>;
%template(UInt8Type) Types::type_info<uint8_t>;
%template(UInt16Type) Types::type_info<uint16_t>;
%template(UInt32Type) Types::type_info<uint32_t>;
%template(UInt64Type) Types::type_info<uint64_t>;
%template(FloatType) Types::type_info<float>;
%template(DoubleType) Types::type_info<double>;
//%template(BoolType) Types::type_info<bool>;

/* Enable callback functions for step, exit and init through the use of "director" which allows Python -> C and C-> Python in callback.
 * FGPU2 supports callback or function pointers so no special tricks are needed. 
 * To prevent raw pointer functions being exposed in Python these are ignored so only the callback versions are accessible.
 */
%feature("director") HostFunctionCallback;
%feature("director") HostFunctionConditionCallback;
%include "flamegpu/runtime/HostFunctionCallback.h"
/* Rather than input a header with lots of other function pointer stuff just inline declare the required enum */
enum FLAME_GPU_CONDITION_RESULT { CONTINUE, EXIT };

// Disable non RTC function and function condition set methods
%ignore AgentDescription::newFunction;
%ignore AgentFunctionDescription::getFunctionPtr;
%ignore AgentFunctionDescription::setFunctionCondition;
%ignore AgentFunctionDescription::getConditionPtr;

// Disable functions which allow raw C pointers in favour of callback objects
%ignore ModelDescription::addInitFunction;
%ignore ModelDescription::addStepFunction;
%ignore ModelDescription::addExitFunction;
%ignore ModelDescription::addExitCondition;
%ignore LayerDescription::addHostFunction;

/* Define ModelData and EnvironmentManager size type (as both are internal to the classes which are not a generated swig object) */
namespace ModelData{
    typedef unsigned int size_type;
}
namespace EnvironmentManager{
    typedef unsigned int size_type;
}

/* SWIG header includes used to generate wrappers */
%include "flamegpu/model/ModelDescription.h"
%include "flamegpu/model/AgentDescription.h"
%include "flamegpu/model/AgentFunctionDescription.h"
%include "flamegpu/model/EnvironmentDescription.h"
%include "flamegpu/model/LayerDescription.h"

%include "flamegpu/model/SubModelDescription.h"
%include "flamegpu/model/SubAgentDescription.h"
%include "flamegpu/model/SubEnvironmentDescription.h"

%include "flamegpu/pop/AgentPopulation.h"
%include "flamegpu/pop/AgentInstance.h"

/* Include Simulation and CUDASimulation */
%feature("flatnested");     // flat nested on to ensure Config is included
%rename (CUDASimulation_Config) CUDASimulation::Config;
%rename (Simulation_Config) Simulation::Config;
%include <argcargv.i>                                           // Include and apply to swig library healer for processing argc and argv values
%apply (int ARGC, char **ARGV) { (int argc, const char **) }    // This is required for CUDASimulation.initialise() 
%include "flamegpu/sim/Simulation.h"
%include "flamegpu/gpu/CUDASimulation.h"
%feature("flatnested", ""); // flat nested off

//%include "flamegpu/runtime/flamegpu_device_api.h"
%ignore VarOffsetStruct; // not required but defined in flamegpu_host_new_agent_api
%include "flamegpu/runtime/flamegpu_host_api.h"
%include "flamegpu/runtime/flamegpu_host_new_agent_api.h"

%include "flamegpu/runtime/flamegpu_host_agent_api.h"
/* Extend HostAgentInstance to add a templated version of the sum function (with differing return type) with a different name so this can be instantiated */
%extend HostAgentInstance{
    template<typename InT, typename OutT> OutT HostAgentInstance::sumOutT(const std::string& variable) const {
        return $self->sum<InT,OutT>(variable);
    }
}


/* Extend HostRandom to add a templated version of the uniform function with a different name so this can be instantiated 
 * It is required to ingore the orginal defintion of uniform and seperate the two functions to have a distinct name
 */
%include "flamegpu/runtime/utility/HostRandom.cuh"
%ignore HostRandom::uniform;
%extend HostRandom{
    template<typename T> inline T uniformRange(const T& min, const T& max) const {
        return $self->uniform<T>(min, max);
    }

    template<typename T> inline T uniformNoRange() const {
        return $self->uniform<T>();
    }
}

%include "flamegpu/runtime/utility/HostEnvironment.cuh"

%include "flamegpu/runtime/AgentFunction_shim.h"
%include "flamegpu/runtime/AgentFunctionCondition_shim.h"


%feature("flatnested");     // flat nested on
// Ignore some of the internal host classes defined for messaging
%ignore *::Data;
%ignore *::CUDAModelHandler;
%ignore *::MetaData;
%rename (MsgBruteForce_Description) MsgBruteForce::Description;
%rename (MsgSpatial2D_Description) MsgSpatial2D::Description;
%rename (MsgSpatial3D_Description) MsgSpatial3D::Description;
%rename (MsgSpatial3D_MetaData) MsgSpatial3D::MetaData;
%rename (MsgArray_Description) MsgArray::Description;
%rename (MsgArray2D_Description) MsgArray2D::Description;
%rename (MsgArray3D_Description) MsgArray3D::Description;
%rename (MsgBucket_Description) MsgBucket::Description;

%include "flamegpu/runtime/messaging/None.h"
%include "flamegpu/runtime/messaging/None/NoneHost.h"
%include "flamegpu/runtime/messaging/BruteForce.h"
%include "flamegpu/runtime/messaging/BruteForce/BruteForceHost.h"
%include "flamegpu/runtime/messaging/Spatial2D.h"
%include "flamegpu/runtime/messaging/Spatial2D/Spatial2DHost.h"
%include "flamegpu/runtime/messaging/Spatial3D.h"
%include "flamegpu/runtime/messaging/Spatial3D/Spatial3DHost.h"
%include "flamegpu/runtime/messaging/Array.h"
%include "flamegpu/runtime/messaging/Array/ArrayHost.h"
%include "flamegpu/runtime/messaging/Array2D.h"
%include "flamegpu/runtime/messaging/Array2D/Array2DHost.h"
%include "flamegpu/runtime/messaging/Array3D.h"
%include "flamegpu/runtime/messaging/Array3D/Array3DHost.h"
%include "flamegpu/runtime/messaging/Bucket.h"
%include "flamegpu/runtime/messaging/Bucket/BucketHost.h"
%feature("flatnested", "");     // flat nested off




// Instantiate template versions of agent functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, AgentDescription::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariableArray, AgentDescription::newVariableArray)

// Instantiate template versions of host agent functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(setVariable, AgentInstance::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE(setVariableArray, AgentInstance::setVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE(getVariable, AgentInstance::getVariable)
TEMPLATE_VARIABLE_INSTANTIATE(getVariableArray, AgentInstance::getVariableArray)

// Instantiate template versions of host agent instance functions from the API
// Not currently supported: custom reductions, transformations or histograms
TEMPLATE_VARIABLE_INSTANTIATE(sort, HostAgentInstance::sort)
TEMPLATE_VARIABLE_INSTANTIATE(count, HostAgentInstance::count)
TEMPLATE_VARIABLE_INSTANTIATE(min, HostAgentInstance::min)
TEMPLATE_VARIABLE_INSTANTIATE(max, HostAgentInstance::max)
TEMPLATE_SUM_INSTANTIATE(HostAgentInstance)

// Instantiate template versions of host environment functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(getProperty, HostEnvironment::getProperty)
TEMPLATE_VARIABLE_INSTANTIATE(getPropertyArray, HostEnvironment::getPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE(setProperty, HostEnvironment::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyArray, HostEnvironment::setPropertyArray)

// Instantiate template versions of host agent functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(getVariable, FLAMEGPU_HOST_NEW_AGENT_API::getVariable)
TEMPLATE_VARIABLE_INSTANTIATE(getVariableArray, FLAMEGPU_HOST_NEW_AGENT_API::getVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE(setVariable, FLAMEGPU_HOST_NEW_AGENT_API::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE(setVariableArray, FLAMEGPU_HOST_NEW_AGENT_API::setVariableArray)


// Instantiate template versions of environment description functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(newProperty, EnvironmentDescription::newProperty)
TEMPLATE_VARIABLE_INSTANTIATE(newPropertyArray, EnvironmentDescription::newPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE(getProperty, EnvironmentDescription::getProperty)
TEMPLATE_VARIABLE_INSTANTIATE(getPropertyArray, EnvironmentDescription::getPropertyArray)
//TEMPLATE_VARIABLE_INSTANTIATE(getPropertyAt, EnvironmentDescription::getPropertyArrayAtIndex)
TEMPLATE_VARIABLE_INSTANTIATE(setProperty, EnvironmentDescription::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyArray, EnvironmentDescription::setPropertyArray)


// Instantiate template versions of new and get message types from the API
%template(newMessageBruteForce) ModelDescription::newMessage<MsgBruteForce>;
%template(newMessageSpatial2D) ModelDescription::newMessage<MsgSpatial2D>;
%template(newMessageSpatial3D) ModelDescription::newMessage<MsgSpatial3D>;
%template(newMessageArray) ModelDescription::newMessage<MsgArray>;
%template(newMessageArray2D) ModelDescription::newMessage<MsgArray2D>;
%template(newMessageArray3D) ModelDescription::newMessage<MsgArray3D>;
%template(newMessageBucket) ModelDescription::newMessage<MsgBucket>;

%template(getMessageBruteForce) ModelDescription::getMessage<MsgBruteForce>;
%template(getMessageSpatial2D) ModelDescription::getMessage<MsgSpatial2D>;
%template(getMessageSpatial3D) ModelDescription::getMessage<MsgSpatial3D>;
%template(getMessageArray) ModelDescription::getMessage<MsgArray>;
%template(getMessageArray2D) ModelDescription::getMessage<MsgArray2D>;
%template(getMessageArray3D) ModelDescription::getMessage<MsgArray3D>;
%template(getMessageBucket) ModelDescription::getMessage<MsgBucket>;

// Instantiate template versions of message functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgBruteForce::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgSpatial2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgSpatial3D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray3D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgBucket::Description::newVariable)



// Instantiate template versions of host random functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(uniform, HostRandom::uniformNoRange)
TEMPLATE_VARIABLE_INSTANTIATE_INTS(uniform, HostRandom::uniformRange)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(normal, HostRandom::normal)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(logNormal, HostRandom::logNormal)






