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
%include <std_list.i>

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
#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/HostFunctionCallback.h"
%}


/* Custom Iterator to Swigify iterable types (RunPlanVec, AgentVector) */
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
         str_str = std::string("(") + type + ") " + msg;
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

/* Include AgentVector/AgentView/AgentInstance */
// Disable functions which use C++ iterators/type_index
%ignore AgentVector::const_iterator;
%ignore AgentVector::const_reverse_iterator;
%ignore AgentVector::iterator;
%ignore AgentVector::const_iterator;
%ignore AgentVector::reverse_iterator;
%ignore AgentVector::const_reverse_iterator;
%ignore AgentVector::begin;
%ignore AgentVector::cbegin;
%ignore AgentVector::end;
%ignore AgentVector::cend;
%ignore AgentVector::rbegin;
%ignore AgentVector::crbegin;
%ignore AgentVector::rend;
%ignore AgentVector::crend;
%ignore AgentVector::insert;
%ignore AgentVector::erase;
%ignore AgentVector::getVariableType;
%ignore AgentVector::getVariableMetaData;
%ignore AgentVector::data;
%rename(insert) AgentVector::py_insert; 
%rename(erase) AgentVector::py_erase; 

// Extend AgentVector so that it is python iterable
%extend AgentVector {
%pythoncode {
    def __iter__(self):
        return FLAMEGPUIterator(self)
    def __len__(self):
        return self.size()
}
    AgentVector::Agent AgentVector::__getitem__(const int &index) {
        if (index >= 0)
            return $self->operator[](index);
        return $self->operator[]($self->size() + index);
    }
    void AgentVector::__setitem__(const AgentVector::size_type &index, const AgentVector::Agent &value) {
        $self->operator[](index).setData(value);
    }
}

%include "flamegpu/pop/AgentVector.h"
%include "flamegpu/pop/AgentVector_Agent.h"
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

// Manually map all template type redirections defined by AgentLoggingConfig
%apply double { sum_input_t<float>::result_t, sum_input_t<double>::result_t }
%apply uint64_t { sum_input_t<char>::result_t, sum_input_t<uint8_t>::result_t, sum_input_t<uint16_t>::result_t, sum_input_t<uint32_t>::result_t, sum_input_t<uint64_t>::result_t }
%apply int64_t { sum_input_t<int8_t>::result_t, sum_input_t<int16_t>::result_t, sum_input_t<int32_t>::result_t, sum_input_t<int64_t>::result_t }

// Include logging implementations
%ignore flamegpu_internal;
%include "flamegpu/sim/LoggingConfig.h"
%include "flamegpu/sim/AgentLoggingConfig.h"
%include "flamegpu/sim/LogFrame.h"
%template(LogFrameList) std::list<LogFrame>;

// Extend RunPlanVec so that it is python iterable
%extend RunPlanVec {
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
   RunPlan &RunPlanVec::__getitem__(const int &index) {
        if (index >= 0)
            return $self->operator[](index);
        return $self->operator[]($self->size() + index);
   }
   void RunPlanVec::__setitem__(const size_t &index, RunPlan &value) {
        $self->operator[](index) = value;
   }
}

// Include ensemble implementations
%include "flamegpu/sim/RunPlan.h"
%include "flamegpu/sim/RunPlanVec.h"
%feature("flatnested");     // flat nested on to ensure Config is included
%rename (CUDAEnsembleConfig) CUDAEnsemble::EnsembleConfig;
%include "flamegpu/gpu/CUDAEnsemble.h"
%feature("flatnested", ""); // flat nested off
%template(RunLogVec) std::vector<RunLog>;


// Instantiate template versions of agent functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, AgentDescription::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariableArray, AgentDescription::newVariableArray)

// Instantiate template versions of AgentVector_Agent/AgentInstance from the API
TEMPLATE_VARIABLE_INSTANTIATE(setVariable, AgentVector_Agent::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE(setVariableArray, AgentVector_Agent::setVariableArray)
TEMPLATE_VARIABLE_INSTANTIATE(getVariable, AgentVector_Agent::getVariable)
TEMPLATE_VARIABLE_INSTANTIATE(getVariableArray, AgentVector_Agent::getVariableArray)

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

// Instantiate template versions of RunPlan functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(setProperty, RunPlan::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyArray, RunPlan::setPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE(getProperty, RunPlan::getProperty)
TEMPLATE_VARIABLE_INSTANTIATE(getPropertyArray, RunPlan::getPropertyArray)

// Instantiate template versions of RunPlanVec functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(setProperty, RunPlanVec::setProperty)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyArray, RunPlanVec::setPropertyArray)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyUniformDistribution, RunPlanVec::setPropertyUniformDistribution)
TEMPLATE_VARIABLE_INSTANTIATE(setPropertyUniformRandomDistribution, RunPlanVec::setPropertyUniformRandom)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(setPropertyNormalRandomDistribution, RunPlanVec::setPropertyNormalRandom)
TEMPLATE_VARIABLE_INSTANTIATE_FLOATS(setPropertyLogNormalRandomDistribution, RunPlanVec::setPropertyLogNormalRandom)

// Instantiate template versions of AgentLoggingConfig functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(logMean, AgentLoggingConfig::logMean)
TEMPLATE_VARIABLE_INSTANTIATE(logMin, AgentLoggingConfig::logMin)
TEMPLATE_VARIABLE_INSTANTIATE(logMax, AgentLoggingConfig::logMax)
TEMPLATE_VARIABLE_INSTANTIATE(logStandardDev, AgentLoggingConfig::logStandardDev)
TEMPLATE_VARIABLE_INSTANTIATE(logSum, AgentLoggingConfig::logSum)

// Instantiate template versions of LogFrame functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(getEnvironmentProperty, LogFrame::getEnvironmentProperty)
TEMPLATE_VARIABLE_INSTANTIATE(getEnvironmentPropertyArray, LogFrame::getEnvironmentPropertyArray)

// Instantiate template versions of AgentLogFrame functions from the API
TEMPLATE_VARIABLE_INSTANTIATE(getMin, AgentLogFrame::getMin)
TEMPLATE_VARIABLE_INSTANTIATE(getMax, AgentLogFrame::getMax)
TEMPLATE_VARIABLE_INSTANTIATE(getSum, AgentLogFrame::getSum)

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

// Optionally instantiate visualisation classes
#ifdef VISUALISATION
%{
#include "flamegpu/visualiser/visualiser_api.h"
#include "config/Stock.h"
%}
// SWIG is unable to wrap `Color::operator StaticColor()`
// Therefore we manually add two functions to handle the implict conversion
%extend AgentStateVis {
   void AgentStateVis::setColor(const Color &cf) {
        $self->setColor(cf);
   }
}
%extend AgentVis {
   void AgentVis::setColor(const Color &cf) {
        $self->setColor(cf);
   }
}
// Disable functions which use C++
%ignore Palette::const_iterator;
%ignore Palette::begin;
%ignore Palette::end;
%ignore Palette::colors;  // This is protected, i've no idea why SWIG is trying to wrap it
// Extend Palette
%extend Palette {
%pythoncode {
    def __iter__(self):
        return FLAMEGPUIterator(self)
    def __len__(self):
        return self.size()
}
    Color Palette::__getitem__(const int &index) {
        if (index >= 0)
            return $self->operator[](index);
        return $self->operator[]($self->size() + index);
    }
    // Palettes are currently immutable
    //void Palette::__setitem__(const AgentVector::size_type &index, const Color &value) {
    //     $self->operator[](index) = value;
    //}
}
%include "flamegpu/visualiser/AgentStateVis.h"
%include "flamegpu/visualiser/AgentVis.h"
%include "flamegpu/visualiser/LineVis.h"
%include "flamegpu/visualiser/ModelVis.h"
%include "flamegpu/visualiser/StaticModelVis.h"
%include "flamegpu/visualiser/color/Color.h"
%include "flamegpu/visualiser/color/Palette.h"
%include "flamegpu/visualiser/color/DiscreteColor.h"
%include "flamegpu/visualiser/color/StaticColor.h"
%include "flamegpu/visualiser/color/HSVInterpolation.h"
%include "config/Stock.h"
// Manually create the two DiscretColor templates
%template(iDiscreteColor) DiscreteColor<int32_t>;
%template(uDiscreteColor) DiscreteColor<uint32_t>;
// This messes with a define, so must occur after all other files which might check VISUALISATION
// #define VISUALISATION false, still causes #ifdef VISUALISATION as true
// I tried `%inline %{const boolean VISUALISATION = false;%}` but swig didnt like it
#undef VISUALISATION
#define VISUALISATION true
#else
#define VISUALISATION false
#endif

#if defined(SEATBELTS) && SEATBELTS
#undef SEATBELTS
#define SEATBELTS true
#elif defined(SEATBELTS)
#undef SEATBELTS
#define SEATBELTS false
#else
#define SEATBELTS false
#endif