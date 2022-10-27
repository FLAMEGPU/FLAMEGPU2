%begin %{
// define SWIG_PYTHON_INTERPRETER_NO_DEBUG on windows debug builds as pythonXX_d is not packaged unless built from source
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

// Suppress known warnings which do not need to be resolved.
// Nested struct not currently supported
#pragma SWIG nowarn=325
// operator= ignored
#pragma SWIG nowarn=362
// operator++ ignored (not supported by python)
#pragma SWIG nowarn=383
// operator-- ignored (not supported by python)
#pragma SWIG nowarn=384
// Warning 451 Setting a const char * variable may leak memory. Fix is to use a std::string instead?
#pragma SWIG nowarn=451
// LoggingConfig.h:65: Function flamegpu::LoggingConfig::Any(ReductionFn) must have a return type. Ignored. This actually a typedef function pointer?
#pragma SWIG nowarn=504


// Other warnings that could be suppressed but we should probably resolve.

// Identifier redefined
#pragma SWIG nowarn=302
// Specialization of non-template
#pragma SWIG nowarn=317
// operator[] ignored (consider using %extend)
#pragma SWIG nowarn=389
// must be defined before it is used as a base class.
#pragma SWIG nowarn=401
// Template 'X' was already wrapped
#pragma SWIG nowarn=404
// Overloaded method X effectively ignored
#pragma SWIG nowarn=509


// name the module, and enabled directors for callback functions.
%module(directors="1") pyflamegpu

// Insert the api header and using namespace directive into the _wrap code.
%{
// Include the main library header, that should subsequently make all other required (public) headers available.
#include "flamegpu/flamegpu.h"
// #include "flamegpu/runtime/HostFunctionCallback.h"
using namespace flamegpu; // @todo - is this required? Ideally it shouldn't be, but swig just dumps stuff into the global namespace. 
%}

// Expand SWIG support for the standard library
%include <stl.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_unordered_map.i>
%include <std_map.i>
%include <std_array.i>
%include <std_list.i>
%include <std_set.i>
%include <std_pair.i>
%include <stdint.i>

// argc/argv support
%include <argcargv.i> 

// Swig exception support
%include "exception.i"

// Enable the use of argc/argv
%apply (int ARGC, char **ARGV) { (int argc, const char **) }


// Macros and Templates replated to types.

// Instantiate the vector types used in the templated variable functions. Types must match those in TEMPLATE_VARIABLE_INSTANTIATE and TEMPLATE_VARIABLE_INSTANTIATE_N macros
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
//%template(DoubleVector) std::vector<double>;

// Instantiate the set type used by CUDAEnsembleConfig.devices
%template(IntSet) std::set<int>;

// Instance the pair type, as returned by HostAgentAPI::meanStandardDeviation
%template(DoublePair) std::pair<double, double>;

/**
 * TEMPLATE_VARIABLE_INSTANTIATE_FLOATS macro
 * Expands for floating point types
 */
%define TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(function, classfunction) 
// float and double
%template(function ## Float) classfunction<float>;
%template(function ## Double) classfunction<double>;
%enddef

// Array version, passing default 2nd template arg 0
%define TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_FLOATS(function, classfunction) 
// float and double
%template(function ## Float) classfunction<float, 0>;
%template(function ## Double) classfunction<double, 0>;
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

// Array version, passing default 2nd template arg 0
%define TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_INTS(function, classfunction) 
// signed ints
%template(function ## Int16) classfunction<int16_t, 0>;
%template(function ## Int32) classfunction<int32_t, 0>;
%template(function ## Int64) classfunction<int64_t, 0>;
// unsigned ints
%template(function ## UInt16) classfunction<uint16_t, 0>;
%template(function ## UInt32) classfunction<uint32_t, 0>;
%template(function ## UInt64) classfunction<uint64_t, 0>;
// default int types
%template(function ## Int) classfunction<int, 0>;
%template(function ## UInt) classfunction<unsigned int, 0>;
%enddef

/**
 * TEMPLATE_VARIABLE_INSTANTIATE macro
 * Given a function name and a class::function specifier, this macro instanciates a typed version of the function for a set of basic types. 
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

// Array version, passing default 2nd template arg 0
%define TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function, classfunction) 
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_FLOATS(function, classfunction)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_INTS(function, classfunction)
// char types
%template(function ## Int8) classfunction<int8_t, 0>;
%template(function ## UInt8) classfunction<uint8_t, 0>;
%template(function ## Char) classfunction<char, 0>;
%template(function ## UChar) classfunction<unsigned char, 0>;
// bool type (not supported causes error)
//%template(function ## Bool) classfunction<bool, 0>;
%enddef

/**
 * Special case of the macro for message types
 * This also maps ID to id_t, this should be synonymous with UInt/unsigned int
 */
%define TEMPLATE_VARIABLE_INSTANTIATE_ID(function, classfunction)
TEMPLATE_VARIABLE_INSTANTIATE(function, classfunction) 
%template(function ## ID) classfunction<flamegpu::id_t>;
%enddef

// Array version, passing default 2nd template arg 0
%define TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(function, classfunction)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function, classfunction) 
%template(function ## ID) classfunction<flamegpu::id_t, 0>;
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

/* Instantiate array types, these are required by message types when setting dimensions. */
%template(UIntArray2) std::array<unsigned int, 2>;
%template(UIntArray3) std::array<unsigned int, 3>;

/* Create some type objects to obtain sizes and type info of flamegpu2 basic types.
 * It is not required to demangle type names. These can be used to compare directly with the type_index 
 * returned by the flamegpu2 library
 */
%nodefaultctor std::type_index;
%inline %{
    namespace Types {
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

// Add custom python code for an iterator class, needed when swigifying iterables.
%pythoncode %{
class FLAMEGPUIterator(object):

    def __init__(self, pointerToVector):
        self.pointerToVector = pointerToVector
        self.index = -1

    def __next__(self):
        self.index += 1
        if self.index < len(self.pointerToVector):
            return self.pointerToVector[self.index]
        else:
            raise StopIteration
%}


// Exception handling.
/** Exception handling
 * FLAMEGPURuntimeException class is a wrapper class to replace specific instances of FLAMEGPUException. It is constructed with a error mesage and type which can be queried to give the original error and the original exception class.
 * The FLAMEGPURuntimeException is constructed form the original FLAMEGPUException in the handler and then translated into the python version.
 * The approach avoids having to wrap every exception class which extends FLAMEGPUException
 */
%exceptionclass FLAMEGPURuntimeException;
// It was hoped that the following would provide a nice repr implementation of the Python FLAMEGPURuntimeException objects. It does not.
//%feature("python:slot", "tp_str", functype="reprfunc") FLAMEGPURuntimeException::what
%inline %{
class FLAMEGPURuntimeException : public std::exception {
 public:
     FLAMEGPURuntimeException(std::string message, std::string type) {
         err_message = message;
         type_str = type;
         str_str = std::string("(") + type + ") " + message;
     }
     const char* what() const noexcept {
         return err_message.c_str();
     }
     const char* type() const {
         return type_str.c_str();
     }
     const char* __str__() const {
         return str_str.c_str();
     }
 protected:
    std::string err_message;
    std::string type_str;
    std::string str_str;
};
%}

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
    catch (flamegpu::exception::FLAMEGPUException& e) {
        FLAMEGPURuntimeException *except = new FLAMEGPURuntimeException(std::string(e.what()), std::string(e.exception_type()));
        PyObject *err = SWIG_NewPointerObj(except, SWIGTYPE_p_FLAMEGPURuntimeException, 1);
        SWIG_Python_Raise(err, except.type(), SWIGTYPE_p_FLAMEGPURuntimeException); 
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


// Ignore directives. These go before any %includes. 
// -----------------

// Disable non RTC function and function condition set methods
%ignore flamegpu::AgentDescription::newFunction;
%ignore flamegpu::AgentFunctionDescription::getFunctionPtr;
%ignore flamegpu::AgentFunctionDescription::setFunctionCondition;
%ignore flamegpu::AgentFunctionDescription::getConditionPtr;

// Ignore function which returns something not currently wrapped.
%ignore flamegpu::AgentInterface::getAgentDescription;

// Disable functions which allow raw C pointers in favour of callback objects
%ignore flamegpu::ModelDescription::addInitFunction;
%ignore flamegpu::ModelDescription::addStepFunction;
%ignore flamegpu::ModelDescription::addExitFunction;
%ignore flamegpu::ModelDescription::addExitCondition;
%ignore flamegpu::LayerDescription::addHostFunction;

// Disable functions which use C++ iterators/type_index
%ignore flamegpu::AgentVector::const_iterator;
%ignore flamegpu::AgentVector::const_reverse_iterator;
%ignore flamegpu::AgentVector::iterator;
%ignore flamegpu::AgentVector::const_iterator;
%ignore flamegpu::AgentVector::reverse_iterator;
%ignore flamegpu::AgentVector::const_reverse_iterator;
%ignore flamegpu::AgentVector::begin;
%ignore flamegpu::AgentVector::cbegin;
%ignore flamegpu::AgentVector::end;
%ignore flamegpu::AgentVector::cend;
%ignore flamegpu::AgentVector::rbegin;
%ignore flamegpu::AgentVector::crbegin;
%ignore flamegpu::AgentVector::rend;
%ignore flamegpu::AgentVector::crend;
%ignore flamegpu::AgentVector::insert;
%ignore flamegpu::AgentVector::erase;
%ignore flamegpu::AgentVector::getVariableType;
%ignore flamegpu::AgentVector::getVariableMetaData;
%ignore flamegpu::AgentVector::data;

%ignore flamegpu::VarOffsetStruct; // not required but defined in HostNewAgentAPI

// Disable functions which use C++ iterators/type_index
%ignore flamegpu::DeviceAgentVector_impl::const_iterator;
%ignore flamegpu::DeviceAgentVector_impl::const_reverse_iterator;
%ignore flamegpu::DeviceAgentVector_impl::iterator;
%ignore flamegpu::DeviceAgentVector_impl::const_iterator;
%ignore flamegpu::DeviceAgentVector_impl::reverse_iterator;
%ignore flamegpu::DeviceAgentVector_impl::const_reverse_iterator;
%ignore flamegpu::DeviceAgentVector_impl::begin;
%ignore flamegpu::DeviceAgentVector_impl::cbegin;
%ignore flamegpu::DeviceAgentVector_impl::end;
%ignore flamegpu::DeviceAgentVector_impl::cend;
%ignore flamegpu::DeviceAgentVector_impl::rbegin;
%ignore flamegpu::DeviceAgentVector_impl::crbegin;
%ignore flamegpu::DeviceAgentVector_impl::rend;
%ignore flamegpu::DeviceAgentVector_impl::crend;
%ignore flamegpu::DeviceAgentVector_impl::insert;
%ignore flamegpu::DeviceAgentVector_impl::erase;
%ignore flamegpu::DeviceAgentVector_impl::getVariableType;
%ignore flamegpu::DeviceAgentVector_impl::getVariableMetaData;
%ignore flamegpu::DeviceAgentVector_impl::data;

%ignore flamegpu::HostRandom::uniform;

// RunPlanVector::SetPropertyRandom takes a c++ std::distribution as an argument, so not appropriate for wrapping.
%ignore flamegpu::RunPlanVector::setPropertyRandom;

// Ignore const'd accessors for configuration structs, which were mutable in python.
%ignore flamegpu::CUDASimulation::getCUDAConfig;
%ignore flamegpu::CUDAEnsemble::getConfig;
%ignore flamegpu::Simulation::getSimulationConfig; // This doesn't currently exist

// Ignore the detail namespace, as it's not intended to be user-facing
%ignore flamegpu::detail;

// Do not provide the FLAMEGPU_VERSION macro, instead just the pyflamegpu.VERSION* variants.
%ignore FLAMEGPU_VERSION;

// Ignores for nested classes, where flatnested is enabled. 
%feature("flatnested"); // flat nested on
    // Ignore some of the internal host classes defined for messaging
    // In the future should these be in the detail namespace which could globally be ignored? // @todo
    %ignore *::Data;
    %ignore *::CUDAModelHandler;
    %ignore *::MetaData;

%feature("flatnested", ""); // flat nested off

// Rename directives. These go before any %includes
// -----------------

%rename(insert) flamegpu::AgentVector::py_insert; 
%rename(erase) flamegpu::AgentVector::py_erase; 

%rename(insert) flamegpu::DeviceAgentVector_impl::py_insert; 
%rename(erase) flamegpu::DeviceAgentVector_impl::py_erase; 

// Renames which require flatnested, as swig/python does not support nested classes.
%feature("flatnested");     // flat nested on to ensure Config is included
    %rename (CUDASimulation_Config) flamegpu::CUDASimulation::Config;
    %rename (Simulation_Config) flamegpu::Simulation::Config;

    %rename (MessageBruteForce_Description) flamegpu::MessageBruteForce::Description;
    %rename (MessageSpatial2D_Description) flamegpu::MessageSpatial2D::Description;
    %rename (MessageSpatial3D_Description) flamegpu::MessageSpatial3D::Description;
    %rename (MessageSpatial3D_MetaData) flamegpu::MessageSpatial3D::MetaData;
    %rename (MessageArray_Description) flamegpu::MessageArray::Description;
    %rename (MessageArray2D_Description) flamegpu::MessageArray2D::Description;
    %rename (MessageArray3D_Description) flamegpu::MessageArray3D::Description;
    %rename (MessageBucket_Description) flamegpu::MessageBucket::Description;

    %rename (CUDAEnsembleConfig) flamegpu::CUDAEnsemble::EnsembleConfig;
%feature("flatnested", ""); // flat nested off

// Director features. These go before the %includes.
// -----------------
/* Enable callback functions for step, exit and init through the use of "director" which allows Python -> C and C-> Python in callback.
 * FLAMEGPU2 supports callback or function pointers so no special tricks are needed. 
 * To prevent raw pointer functions being exposed in Python these are ignored so only the callback versions are accessible.
 */
%feature("director") flamegpu::HostFunctionCallback;
%feature("director") flamegpu::HostFunctionConditionCallback;

// Apply type mappings go before %includes.
// -----------------

// Manually map all template type redirections defined by AgentLoggingConfig
%apply double { flamegpu::sum_input_t<float>::result_t, flamegpu::sum_input_t<double>::result_t }
%apply uint64_t { flamegpu::sum_input_t<char>::result_t, flamegpu::sum_input_t<uint8_t>::result_t, flamegpu::sum_input_t<uint16_t>::result_t, flamegpu::sum_input_t<uint32_t>::result_t, flamegpu::sum_input_t<uint64_t>::result_t }
%apply int64_t { flamegpu::sum_input_t<int8_t>::result_t, flamegpu::sum_input_t<int16_t>::result_t, flamegpu::sum_input_t<int32_t>::result_t, flamegpu::sum_input_t<int64_t>::result_t }

// Value wrappers also go before includes.
// -----------------
// %feature("valuewrapper") flamegpu::DeviceAgentVector; // @todo - this doesn't appear to be required.
 
// Enums / type definitions.
// -----------------

// Define ModelData and EnvironmentManager size type (as both are internal to the classes which are not a generated swig object)
// Long term these should probable be either made publicly available or not used where they are not available.
namespace flamegpu {
namespace ModelData{
    typedef unsigned int size_type;
}
namespace EnvironmentManager{
    typedef unsigned int size_type;
}
}

// Forward declare some classes as necessary.
// -----------------
// This is required where there are circular dependencies and how swig doesn't #include things. Instead, forward declare the class within the namespace that is otherwise #included.

namespace flamegpu { 
class ModelDescription;  // For DependencyGraph circular dependency. 
}

// If visualisation is enabled, then CUDASimulation provides access to the visualisation class. This requires a forward declaraiton to place it in the correct namespace. 
#ifdef VISUALISATION
namespace flamegpu {
namespace visualiser {
class ModelVis;
} // namespace visualiser
}  // namespace flamegpu
#endif

// %includes for classes to wrap. 
// -----------------
// A number of typedefs are not placed in the namespace, but they are currently unused anyway. 
// SWIGTYPE_p_FLAMEGPURuntimeException - swig only, doesn't need to be namespaced? 

%include "flamegpu/defines.h" // Provides flamegpu::id_t amongst others.
%include "flamegpu/version.h" // provides FLAMEGPU_VERSION etc
%include "flamegpu/runtime/HostAPI_macros.h" // Used in LayerDesc, LayerData, HostFuncDesc

%include "flamegpu/sim/AgentInterface.h"

%include "flamegpu/runtime/HostFunctionCallback.h"

%feature("flatnested");     // flat nested on
%include "flamegpu/runtime/messaging/MessageNone.h"
%include "flamegpu/runtime/messaging/MessageNone/MessageNoneHost.h"
%include "flamegpu/runtime/messaging/MessageBruteForce.h"
%include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h"  // must be before ModelDesc, AgentFunctionDescription and more complex messaging types.
%include "flamegpu/runtime/messaging/MessageSpatial2D.h"
%include "flamegpu/runtime/messaging/MessageSpatial2D/MessageSpatial2DHost.h"
%include "flamegpu/runtime/messaging/MessageSpatial3D.h"
%include "flamegpu/runtime/messaging/MessageSpatial3D/MessageSpatial3DHost.h"
%include "flamegpu/runtime/messaging/MessageArray.h"
%include "flamegpu/runtime/messaging/MessageArray/MessageArrayHost.h"
%include "flamegpu/runtime/messaging/MessageArray2D.h"
%include "flamegpu/runtime/messaging/MessageArray2D/MessageArray2DHost.h"
%include "flamegpu/runtime/messaging/MessageArray3D.h"
%include "flamegpu/runtime/messaging/MessageArray3D/MessageArray3DHost.h"
%include "flamegpu/runtime/messaging/MessageBucket.h"
%include "flamegpu/runtime/messaging/MessageBucket/MessageBucketHost.h"
%feature("flatnested", "");     // flat nested off

%include "flamegpu/model/DependencyNode.h"
%include "flamegpu/model/DependencyGraph.h"

%include "flamegpu/model/EnvironmentDescription.h"
%include "flamegpu/model/ModelDescription.h"
%include "flamegpu/model/HostFunctionDescription.h"
%include "flamegpu/model/AgentDescription.h"
%include "flamegpu/model/AgentFunctionDescription.h"
%include "flamegpu/model/LayerDescription.h"
%include "flamegpu/model/SubModelDescription.h"
%include "flamegpu/model/SubAgentDescription.h"
%include "flamegpu/model/SubEnvironmentDescription.h"

%include "flamegpu/runtime/utility/RandomManager.cuh"

// Include Simulation and CUDASimulation
%feature("flatnested");     // flat nested on to ensure Config is included
%include "flamegpu/sim/Simulation.h"
%include "flamegpu/gpu/CUDASimulation.h"
%feature("flatnested", ""); // flat nested off

%feature("flatnested");     // flat nested on to ensure Config is included
%include "flamegpu/gpu/CUDAEnsemble.h"
%feature("flatnested", ""); // flat nested off

%include "flamegpu/runtime/AgentFunction_shim.cuh"
%include "flamegpu/runtime/AgentFunctionCondition_shim.cuh"

// These are essentially nested classes that have been split out. 
%include "flamegpu/pop/AgentVector_Agent.h"
%include "flamegpu/pop/AgentVector.h"
%include "flamegpu/pop/AgentInstance.h"
%include "flamegpu/pop/DeviceAgentVector_impl.h"
%include "flamegpu/pop/DeviceAgentVector.h"

// Must wrap these prior to HostAPI where they are used to avoid issues with no default constructors etc.
%include "flamegpu/runtime/utility/HostRandom.cuh"

%nodefaultctor flamegpu::HostMacroProperty_swig;
%include "flamegpu/runtime/utility/HostMacroProperty.cuh"
%include "flamegpu/runtime/utility/HostEnvironment.cuh"

%include "flamegpu/runtime/HostNewAgentAPI.h"
%include "flamegpu/runtime/HostAgentAPI.cuh"
%include "flamegpu/runtime/HostAPI.h" 

// Include logging implementations
%include "flamegpu/sim/LoggingConfig.h"
%include "flamegpu/sim/AgentLoggingConfig.h"
%include "flamegpu/sim/AgentLoggingConfig_SumReturn.h"
%include "flamegpu/sim/LogFrame.h"  // Includes RunLog. 

// Include ensemble implementations
%include "flamegpu/sim/RunPlan.h"
%include "flamegpu/sim/RunPlanVector.h"

// Include  cleanup utility method
%include "flamegpu/util/cleanup.h"

// %extend classes go after %includes, but before tempalates (that use them)
// -----------------

// Extend HostAgentAPI to add a templated version of the sum function (with differing return type) with a different name so this can be instantiated
%extend flamegpu::HostAgentAPI{
    template<typename InT, typename OutT> OutT flamegpu::HostAgentAPI::sumOutT(const std::string& variable) const {
        return $self->sum<InT,OutT>(variable);
    }
}

// Extend AgentVector so that it is python iterable
%extend flamegpu::AgentVector {
    %pythoncode {
        def __iter__(self):
            return FLAMEGPUIterator(self)
        def __len__(self):
            return self.size()
    }
    flamegpu::AgentVector::Agent flamegpu::AgentVector::__getitem__(const int index) {
        if (index >= 0)
            return $self->operator[](index);
        return $self->operator[]($self->size() + index);
    }
    void flamegpu::AgentVector::__setitem__(const flamegpu::size_type &index, const flamegpu::AgentVector::Agent &value) {
        $self->operator[](index).setData(value);
    }
}
/* Extend HostRandom to add a templated version of the uniform function with a different name so this can be instantiated 
 * It is required to ingore the original defintion of uniform and separate the two functions to have a distinct name
 */
%extend flamegpu::HostRandom{
    template<typename T> inline T uniformRange(const T min, const T max) const {
        return $self->uniform<T>(min, max);
    }

    template<typename T> inline T uniformNoRange() const {
        return $self->uniform<T>();
    }
}
// Extend RunPlanVector so that it is python iterable
%extend flamegpu::RunPlanVector {
%pythoncode {
    def __iter__(self):
        return FLAMEGPUIterator(self)
    def __len__(self):
        return self.size()

    def insert(self, i, x):
        if isinstance(i, int): # "insert" is used as if the vector is a Python list
            self.insert(self, self.begin() + i, x)
        else: # "insert" is used as if the vector is a native C++ container
            return self.insert(self, i, x)
   }
   flamegpu::RunPlan &flamegpu::RunPlanVector::__getitem__(const int index) {
        if (index >= 0)
            return $self->operator[](index);
        return $self->operator[]($self->size() + index);
   }
   void RunPlanVector::__setitem__(const size_t index, flamegpu::RunPlan &value) {
        $self->operator[](index) = value;
   }
}

// Extend flamegpu::DeviceAgentVector so that it is python iterable
%extend flamegpu::DeviceAgentVector_impl {
    %pythoncode {
        def __iter__(self):
            return FLAMEGPUIterator(self)
        def __len__(self):
            return self.size()
    }
    flamegpu::DeviceAgentVector_impl::Agent flamegpu::DeviceAgentVector_impl::__getitem__(const int index) {
        if (index >= 0)
            return $self->operator[](index);
        return $self->operator[]($self->size() + index);
    }
    void flamegpu::DeviceAgentVector_impl::__setitem__(const size_type index, const Agent &value) {
        $self->operator[](index).setData(value);
    }
}

// Template expansions. Go after the %include and %extension
// -----------------

// DependencyNode template instantiations
%template(dependsOn) flamegpu::DependencyNode::dependsOn<flamegpu::DependencyNode>;

%template(StepLogFrameList) std::list<flamegpu::StepLogFrame>;
%template(RunLogVec) std::vector<flamegpu::RunLog>;
 
// Instantiate template versions of agent functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::AgentDescription::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::AgentDescription::newVariableArray)

// Instantiate template versions of AgentVector_Agent/AgentInstance from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(setVariable, flamegpu::AgentVector_Agent::setVariable)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setVariable, flamegpu::AgentVector_Agent::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setVariableArray, flamegpu::AgentVector_Agent::setVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getVariable, flamegpu::AgentVector_Agent::getVariable)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(getVariable, flamegpu::AgentVector_Agent::getVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getVariableArray, flamegpu::AgentVector_Agent::getVariableArray)

TEMPLATE_VARIABLE_INSTANTIATE_ID(setVariable, flamegpu::AgentInstance::setVariable)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setVariable, flamegpu::AgentInstance::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setVariableArray, flamegpu::AgentInstance::setVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getVariable, flamegpu::AgentInstance::getVariable)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(getVariable, flamegpu::AgentInstance::getVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getVariableArray, flamegpu::AgentInstance::getVariableArray)

// Instantiate template versions of host agent instance functions from the API
// Not currently supported: custom reductions, transformations or histograms
TEMPLATE_VARIABLE_INSTANTIATE(sort, flamegpu::HostAgentAPI::sort)
TEMPLATE_VARIABLE_INSTANTIATE(count, flamegpu::HostAgentAPI::count)
TEMPLATE_VARIABLE_INSTANTIATE(min, flamegpu::HostAgentAPI::min)
TEMPLATE_VARIABLE_INSTANTIATE(max, flamegpu::HostAgentAPI::max)
TEMPLATE_SUM_INSTANTIATE(flamegpu::HostAgentAPI)
TEMPLATE_VARIABLE_INSTANTIATE(meanStandardDeviation, flamegpu::HostAgentAPI::meanStandardDeviation)

// Instantiate template versions of host environment functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(getProperty, flamegpu::HostEnvironment::getProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(getProperty, flamegpu::HostEnvironment::getProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getPropertyArray, flamegpu::HostEnvironment::getPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setProperty, flamegpu::HostEnvironment::setProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setProperty, flamegpu::HostEnvironment::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setPropertyArray, flamegpu::HostEnvironment::setPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getMacroProperty, flamegpu::HostEnvironment::getMacroProperty_swig)

// Instantiate template versions of CUDASimulation environment functions
TEMPLATE_VARIABLE_INSTANTIATE_ID(setEnvironmentProperty, flamegpu::CUDASimulation::setEnvironmentProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setEnvironmentProperty, flamegpu::CUDASimulation::setEnvironmentProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setEnvironmentPropertyArray, flamegpu::CUDASimulation::setEnvironmentPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getEnvironmentProperty, flamegpu::CUDASimulation::getEnvironmentProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(getEnvironmentProperty, flamegpu::CUDASimulation::getEnvironmentProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getEnvironmentPropertyArray, flamegpu::CUDASimulation::getEnvironmentPropertyArray)

// Instance template versions of the HostMacroProperty class
// Extend HostMacroProperty so that it is python iterable
%extend flamegpu::HostMacroProperty_swig {
    %pythoncode {
        def __iter__(self):
            return FLAMEGPUIterator(self)
        def __add__(self, other):
            return self.get() + other;
        def __radd__(self, other):
            return other + self.get();
        def __iadd__(self, other):
            self.set(self.get() + other);
            return self;
        def __sub__(self, other):
            return self.get() - other;
        def __rsub__(self, other):
            return other - self.get();
        def __isub__(self, other):
            self.set(self.get() - other);
            return self;
        def __mul__(self, other):
            return self.get() * other;
        def __rmul__(self, other):
            return other * self.get();
        def __imul__(self, other):
            self.set(self.get() * other);
            return self;
        def __pow__(self, other):
            return self.get() ** other;
        def __rpow__(self, other):
            return other ** self.get();
        def __ipow__(self, other):
            self.set(self.get() ** other);
            return self;
        def __truediv__(self, other):
            return self.get() / other;
        def __rtruediv__(self, other):
            return other / self.get();
        def __itruediv__(self, other):
            try:
                self.set(self.get() / other);
            except:
                raise FLAMEGPURuntimeException("__itruediv__ does not support the used type combination as it would lead to type conversion of the host object.", "unsupported type")
            return self;
        def __floordiv__(self, other):
            return self.get() // other;
        def __rfloordiv__(self, other):
            return other // self.get();
        def __ifloordiv__(self, other):
            self.set(self.get() // other);
            return self;
        def __mod__(self, other):
            return self.get() % other;
        def __rmod__(self, other):
            return other % self.get();
        def __imod__(self, other):
            try:
                self.set(self.get() % other);
            except:
                raise FLAMEGPURuntimeException("__imod__ does not support the used type combination as it would lead to type conversion of the host object.", "unsupported type")
            return self;
    }
}
TEMPLATE_VARIABLE_INSTANTIATE_ID(HostMacroProperty, flamegpu::HostMacroProperty_swig)

// Instantiate template versions of host agent functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(getVariable, flamegpu::HostNewAgentAPI::getVariable)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(getVariable, flamegpu::HostNewAgentAPI::getVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getVariableArray, flamegpu::HostNewAgentAPI::getVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setVariable, flamegpu::HostNewAgentAPI::setVariable)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setVariable, flamegpu::HostNewAgentAPI::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setVariableArray, flamegpu::HostNewAgentAPI::setVariableArray)


// Instantiate template versions of environment description functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(newProperty, flamegpu::EnvironmentDescription::newProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newPropertyArray, flamegpu::EnvironmentDescription::newPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newMacroProperty, flamegpu::EnvironmentDescription::newMacroProperty_swig)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getProperty, flamegpu::EnvironmentDescription::getProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getPropertyArray, flamegpu::EnvironmentDescription::getPropertyArray)
//TEMPLATE_VARIABLE_INSTANTIATE_ID(getPropertyAt, flamegpu::EnvironmentDescription::getPropertyArrayAtIndex)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setProperty, flamegpu::EnvironmentDescription::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setPropertyArray, flamegpu::EnvironmentDescription::setPropertyArray)

// Instantiate template versions of RunPlan functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(setProperty, flamegpu::RunPlan::setProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setProperty, flamegpu::RunPlan::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setPropertyArray, flamegpu::RunPlan::setPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getProperty, flamegpu::RunPlan::getProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(getProperty, flamegpu::RunPlan::getProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getPropertyArray, flamegpu::RunPlan::getPropertyArray)

// Instantiate template versions of RunPlanVector functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(setProperty, flamegpu::RunPlanVector::setProperty)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(setProperty, flamegpu::RunPlanVector::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(setPropertyArray, flamegpu::RunPlanVector::setPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyUniformDistribution, flamegpu::RunPlanVector::setPropertyUniformDistribution)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyUniformRandom, flamegpu::RunPlanVector::setPropertyUniformRandom)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(setPropertyNormalRandom, flamegpu::RunPlanVector::setPropertyNormalRandom)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(setPropertyLogNormalRandom, flamegpu::RunPlanVector::setPropertyLogNormalRandom)

// Instantiate template versions of AgentLoggingConfig functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(logMean, flamegpu::AgentLoggingConfig::logMean)
TEMPLATE_VARIABLE_INSTANTIATE(logMin, flamegpu::AgentLoggingConfig::logMin)
TEMPLATE_VARIABLE_INSTANTIATE(logMax, flamegpu::AgentLoggingConfig::logMax)
TEMPLATE_VARIABLE_INSTANTIATE(logStandardDev, flamegpu::AgentLoggingConfig::logStandardDev)
TEMPLATE_VARIABLE_INSTANTIATE(logSum, flamegpu::AgentLoggingConfig::logSum)

// Instantiate template versions of LogFrame functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(getEnvironmentProperty, flamegpu::LogFrame::getEnvironmentProperty)
TEMPLATE_VARIABLE_INSTANTIATE_ID(getEnvironmentPropertyArray, flamegpu::LogFrame::getEnvironmentPropertyArray)

// Instantiate template versions of AgentLogFrame functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(getMin, flamegpu::AgentLogFrame::getMin)
TEMPLATE_VARIABLE_INSTANTIATE(getMax, flamegpu::AgentLogFrame::getMax)
TEMPLATE_VARIABLE_INSTANTIATE(getSum, flamegpu::AgentLogFrame::getSum)

// Instantiate template versions of new and get message types from the API
%template(newMessageBruteForce) flamegpu::ModelDescription::newMessage<flamegpu::MessageBruteForce>;
%template(newMessageSpatial2D) flamegpu::ModelDescription::newMessage<flamegpu::MessageSpatial2D>;
%template(newMessageSpatial3D) flamegpu::ModelDescription::newMessage<flamegpu::MessageSpatial3D>;
%template(newMessageArray) flamegpu::ModelDescription::newMessage<flamegpu::MessageArray>;
%template(newMessageArray2D) flamegpu::ModelDescription::newMessage<flamegpu::MessageArray2D>;
%template(newMessageArray3D) flamegpu::ModelDescription::newMessage<flamegpu::MessageArray3D>;
%template(newMessageBucket) flamegpu::ModelDescription::newMessage<flamegpu::MessageBucket>;

%template(getMessageBruteForce) flamegpu::ModelDescription::getMessage<MessageBruteForce>;
%template(getMessageSpatial2D) flamegpu::ModelDescription::getMessage<MessageSpatial2D>;
%template(getMessageSpatial3D) flamegpu::ModelDescription::getMessage<MessageSpatial3D>;
%template(getMessageArray) flamegpu::ModelDescription::getMessage<MessageArray>;
%template(getMessageArray2D) flamegpu::ModelDescription::getMessage<MessageArray2D>;
%template(getMessageArray3D) flamegpu::ModelDescription::getMessage<MessageArray3D>;
%template(getMessageBucket) flamegpu::ModelDescription::getMessage<MessageBucket>;


// Instantiate template versions of message functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageBruteForce::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageSpatial2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageSpatial3D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageArray::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageArray2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageArray3D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariable, flamegpu::MessageBucket::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageBruteForce::Description::newVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageSpatial2D::Description::newVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageSpatial3D::Description::newVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageArray::Description::newVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageArray2D::Description::newVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageArray3D::Description::newVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE_ID(newVariableArray, flamegpu::MessageBucket::Description::newVariableArray)

// Instantiate template versions of host random functions from the API
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(uniform, flamegpu::HostRandom::uniformNoRange)
TEMPLATE_VARIABLE_INSTANTIATE_INTS(uniform, flamegpu::HostRandom::uniformRange)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(normal, flamegpu::HostRandom::normal)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(logNormal, flamegpu::HostRandom::logNormal)

// Extend the python to add the pure python class decorators
%pythoncode %{

    from functools import wraps

    def agent_function(func):
        @wraps(func)
        def wrapper():
            # do not allow passthrough (host exection of this function)
            pass
        return wrapper
		
    def agent_function_condition(func):
        @wraps(func)
        def wrapper():
            # do not allow passthrough (host exection of this function)
            pass
        return wrapper

    def device_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # essenitally a passthrough in case the function is also used in python host code
            # passthrough obviously does not support pyflamegpu device functions
            return func(*args, **kwargs)
        
        # create an attribute to identify the wrapped function as having this decorator (without having to parse)
        wrapper.__is_pyflamegpu_device_function = True
        return wrapper
%}



// Include visualisation support if enabled.
#ifdef VISUALISATION
    // Include relevant headers in the generated c++
    // @todo - Need to put the vis repo into a subfolder for more sensible include paths
    %{
        #include "flamegpu/visualiser/visualiser_api.h"
        #include "flamegpu/visualiser/config/Stock.h"
        using namespace flamegpu;
        using namespace flamegpu::visualiser;
    %}

    // Ignore directives. These go before any %includes. 
    // -----------------
    // Disable functions which return std::type_index
    %ignore flamegpu::visualiser::ColorFunction::getAgentVariableRequiredType;
    %ignore flamegpu::visualiser::DiscreteColor::getAgentVariableRequiredType;
    %ignore flamegpu::visualiser::HSVInterpolation::getAgentVariableRequiredType;
    %ignore flamegpu::visualiser::ViridisInterpolation::getAgentVariableRequiredType;
    // Disable functions which use C++
    %ignore flamegpu::visualiser::Color::operator StaticColor;
    %ignore flamegpu::visualiser::Palette::const_iterator;
    %ignore flamegpu::visualiser::Palette::begin;
    %ignore flamegpu::visualiser::Palette::end;
    %ignore flamegpu::visualiser::Palette::colors;  // This is protected, i've no idea why SWIG is trying to wrap it
    // Mark PanelVis as a class where assignment operator is not supported
    %feature("valuewrapper") flamegpu::visualiser::PanelVis;
    // Rename directives. These go before any %includes
    // -----------------
    // Director features. These go before the %includes.
    // -----------------
    // Enums / type definitions.
    // -----------------
    // %includes for classes to wrap. 
    // -----------------
    %include "flamegpu/visualiser/config/Stock.h"
    %include "flamegpu/visualiser/StaticModelVis.h"
    %include "flamegpu/visualiser/AgentStateVis.h"
    %include "flamegpu/visualiser/AgentVis.h"
    %include "flamegpu/visualiser/LineVis.h"
    %include "flamegpu/visualiser/PanelVis.h"
    %include "flamegpu/visualiser/ModelVis.h"
    %include "flamegpu/visualiser/color/Color.h"
    %include "flamegpu/visualiser/color/ColorFunction.h"
    %include "flamegpu/visualiser/color/Palette.h"
    %include "flamegpu/visualiser/color/DiscreteColor.h"
    %include "flamegpu/visualiser/color/StaticColor.h"
    %include "flamegpu/visualiser/color/HSVInterpolation.h"
    %include "flamegpu/visualiser/color/ViridisInterpolation.h"
    // @todo - this probably does need to be wrapped.
    // %extend classes go after %includes.
    // -----------------
    // SWIG is unable to wrap `Color::operator StaticColor()`
    // Therefore we manually add two functions to handle the implicit conversion
    %extend flamegpu::visualiser::AgentStateVis {
    void flamegpu::visualiser::AgentStateVis::setColor(const flamegpu::visualiser::Color &cf) {
            $self->setColor(cf);
    }
    }
    %extend flamegpu::visualiser::AgentVis {
    void flamegpu::visualiser::AgentVis::setColor(const flamegpu::visualiser::Color &cf) {
            $self->setColor(cf);
    }
    }
    // Extend Palette
    %extend flamegpu::visualiser::Palette {
        %pythoncode {
            def __iter__(self):
                return FLAMEGPUIterator(self)
            def __len__(self):
                return self.size()
        }
        flamegpu::visualiser::Color flamegpu::visualiser::Palette::__getitem__(const int index) {
            if (index >= 0)
                return $self->operator[](index);
            return $self->operator[]($self->size() + index);
        }
        // Palettes are currently immutable
        //void Palette::__setitem__(const flamegpu::size_type &index, const Color &value) {
        //     $self->operator[](index) = value;
        //}
    }

    // Template expansions. Go after the %include and extends
    // -----------------
    // Must declare the base class version above before instantiating template
    %template(iColorMap) std::map<int32_t, flamegpu::visualiser::Color>;
    %template(uColorMap) std::map<uint32_t, flamegpu::visualiser::Color>;
    // Manually create the two DiscretColor templates
    %template(iDiscreteColor) flamegpu::visualiser::DiscreteColor<int32_t>;
    %template(uDiscreteColor) flamegpu::visualiser::DiscreteColor<uint32_t>;
    TEMPLATE_VARIABLE_INSTANTIATE_ID(newEnvironmentPropertySlider, flamegpu::visualiser::PanelVis::newEnvironmentPropertySlider)
    TEMPLATE_VARIABLE_INSTANTIATE_ID(newEnvironmentPropertyDrag, flamegpu::visualiser::PanelVis::newEnvironmentPropertyDrag)
    TEMPLATE_VARIABLE_INSTANTIATE_ID(newEnvironmentPropertyInput, flamegpu::visualiser::PanelVis::newEnvironmentPropertyInput)
    TEMPLATE_VARIABLE_INSTANTIATE_INTS(newEnvironmentPropertyToggle, flamegpu::visualiser::PanelVis::newEnvironmentPropertyToggle)
    
    TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(newEnvironmentPropertySlider, flamegpu::visualiser::PanelVis::newEnvironmentPropertySlider)
    TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(newEnvironmentPropertyDrag, flamegpu::visualiser::PanelVis::newEnvironmentPropertyDrag)
    TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_ID(newEnvironmentPropertyInput, flamegpu::visualiser::PanelVis::newEnvironmentPropertyInput)
    TEMPLATE_VARIABLE_ARRAY_INSTANTIATE_INTS(newEnvironmentPropertyToggle, flamegpu::visualiser::PanelVis::newEnvironmentPropertyToggle)

    
    // Redefine the value to ensure it makes it into the python modules
    #undef VISUALISATION
    #define VISUALISATION true
#else 
    // Define in the python module as false.
    #define VISUALISATION false
#endif

// Define pyflamegpu.SEATBELTS as true or false as appropriate, so tests can be disabled / enabled  
#if defined(SEATBELTS) && SEATBELTS
    #undef SEATBELTS
    #define SEATBELTS true
#elif defined(SEATBELTS)
    #undef SEATBELTS
    #define SEATBELTS false
#else
    #define SEATBELTS false
#endif