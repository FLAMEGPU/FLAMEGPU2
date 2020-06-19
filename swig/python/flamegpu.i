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
%include <std_string.i>
%include <std_vector.i>
%include <std_unordered_map.i>
//%include <std_list.i>


%module(directors="1") pyflamegpu



/**
 * TEMPLATE_VARIABLE_ARRAY_INSTANTIATE macro
 * Given a function name and a class::function specifier, this macro instaciates a typed array size version of the function for a set of basic types. 
 * E.g. TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(functionType, SomeClass:function, int) will generate swig typed versions of the function like the following
 *    typedef SomeClass:function<int, 1> functionInt;
 *    typedef SomeClass:function<int, 2> functionInt;
 *    typedef SomeClass:function<int, 3> functionInt;
 *    typedef SomeClass:function<int, 4> functionInt;
 *    ...
 */
%define TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(functionType, classfunction, T) 
%template(functionType) classfunction<T, 1>;
%template(functionType ## A2) classfunction<T, 2>;
%template(functionType ## A3) classfunction<T, 3>;
%template(functionType ## A4) classfunction<T, 4>;
%template(functionType ## A8) classfunction<T, 8>;
%template(functionType ## A16) classfunction<T, 16>;
%template(functionType ## A32) classfunction<T, 32>;
%template(functionType ## A64) classfunction<T, 64>;
%template(functionType ## A128) classfunction<T, 128>;
%template(functionType ## A256) classfunction<T, 256>;
%template(functionType ## A512) classfunction<T, 512>;
%template(functionType ## A1024) classfunction<T, 1024>;
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
// signed ints
%template(function ## Int8) classfunction<int8_t>;
%template(function ## Int16) classfunction<int16_t>;
%template(function ## Int32) classfunction<int32_t>;
%template(function ## Int64) classfunction<int64_t>;
// unsigned ints
%template(function ## UInt8) classfunction<uint8_t>;
%template(function ## UInt16) classfunction<uint16_t>;
%template(function ## UInt32) classfunction<uint32_t>;
%template(function ## UInt64) classfunction<uint64_t>;
// float and double
%template(function ## Float) classfunction<float>;
%template(function ## Double) classfunction<double>;
// default int types
%template(function ## Int) classfunction<int>;
%template(function ## UInt) classfunction<unsigned int>;
%enddef

/**
 * TEMPLATE_VARIABLE_INSTANTIATE_N macro
 * Given a function name and a class::function specifier, this macro instanciates a typed version of the function for a set of basic types AND default array lengths. 
 * See description of TEMPLATE_VARIABLE_INSTANTIATE and TEMPLATE_VARIABLE_ARRAY_INSTANTIATE
 */
%define TEMPLATE_VARIABLE_INSTANTIATE_N(function, classfunction) 
// generate non array versions
TEMPLATE_VARIABLE_INSTANTIATE(function, classfunction)
// signed ints
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Int8, classfunction, int8_t)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Int16, classfunction, int16_t)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Int32, classfunction, int32_t)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Int64, classfunction, int8_t)
// unsigned ints
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## UInt8, classfunction, uint8_t)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## UInt16, classfunction, uint16_t)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## UInt32, classfunction, uint32_t)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## UInt64, classfunction, uint8_t)
// float and double
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Float, classfunction, float)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Double, classfunction, double)
// default int and uint (causes redefintion warning)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## Int, classfunction, int)
TEMPLATE_VARIABLE_ARRAY_INSTANTIATE(function ## UInt, classfunction, unsigned int)
%enddef

/* Compilation header includes */
%{
/* Includes the header in the wrapper code */
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

#include "flamegpu/exception/FGPUException.h"

//#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/runtime/flamegpu_host_agent_api.h"
#include "flamegpu/runtime/messaging.h"


#include "flamegpu/runtime/AgentFunction_shim.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
#include "flamegpu/runtime/HostFunctionCallback.h"
%}

/* Callback functions for step, exit and init */
%feature("director") HostFunctionCallback;
%include "flamegpu/runtime/HostFunctionCallback.h"


/* Disable non RTC function and function condtion set methods */
%ignore AgentDescription::newFunction;
%ignore AgentFunctionDescription::getFunctionPtr;
%ignore AgentFunctionDescription::setFunctionCondition;
%ignore AgentFunctionDescription::getConditionPtr;

/* Disable functions which allow raw C pointers in favour of callback objects */
%ignore ModelDescription::addInitFunction;
%ignore ModelDescription::addStepFunction;
%ignore ModelDescription::addExitFunction;


/* SWIG header includes used to generate wrappers */
%include "flamegpu/model/ModelDescription.h"
%include "flamegpu/model/AgentDescription.h"
%include "flamegpu/model/AgentFunctionDescription.h"
//%include "flamegpu/model/EnvironmentDescription.h"
%include "flamegpu/model/LayerDescription.h"
%include "flamegpu/pop/AgentPopulation.h"
%include "flamegpu/pop/AgentInstance.h"


/* Exceptions */
%varargs (void* null=NULL) CUDAError;
%varargs (void* null=NULL) ReservedName;
%varargs (void* null=NULL) InvalidInputFile;
%varargs (void* null=NULL) InvalidHashList;
%varargs (void* null=NULL) InvalidVarType;
%varargs (void* null=NULL) UnsupportedVarType;
%varargs (void* null=NULL) InvalidStateName;
%varargs (void* null=NULL) InvalidMapEntry;
%varargs (void* null=NULL) InvalidParent;
%varargs (void* null=NULL) InvalidAgentName;
%varargs (void* null=NULL) InvalidMessageName;
%varargs (void* null=NULL) InvalidMessageType;
%varargs (void* null=NULL) InvalidAgent;
%varargs (void* null=NULL) InvalidMessage;
%varargs (void* null=NULL) InvalidAgentVar;
%varargs (void* null=NULL) InvalidVarArrayLen;
%varargs (void* null=NULL) OutOfRangeVarArray;
%varargs (void* null=NULL) InvalidMessageVar;
%varargs (void* null=NULL) InvalidMessageData;
%varargs (void* null=NULL) InvalidMessageSize;
%varargs (void* null=NULL) InvalidCudaAgent;
%varargs (void* null=NULL) InvalidCudaMessage;
%varargs (void* null=NULL) InvalidCudaAgentMapSize;
%varargs (void* null=NULL) InvalidCudaAgentDesc;
%varargs (void* null=NULL) InvalidCudaAgentState;
%varargs (void* null=NULL) InvalidAgentFunc;
%varargs (void* null=NULL) InvalidFuncLayerIndx;
%varargs (void* null=NULL) InvalidPopulationData;
%varargs (void* null=NULL) InvalidMemoryCapacity;
%varargs (void* null=NULL) InvalidOperation;
%varargs (void* null=NULL) InvalidCUDAdevice;
%varargs (void* null=NULL) InvalidCUDAComputeCapability;
%varargs (void* null=NULL) InvalidHostFunc;
%varargs (void* null=NULL) InvalidArgument;
%varargs (void* null=NULL) DuplicateEnvProperty;
%varargs (void* null=NULL) InvalidEnvProperty;
%varargs (void* null=NULL) InvalidEnvPropertyType;
%varargs (void* null=NULL) ReadOnlyEnvProperty;
%varargs (void* null=NULL) EnvDescriptionAlreadyLoaded;
%varargs (void* null=NULL) OutOfMemory;
%varargs (void* null=NULL) CurveException;
%varargs (void* null=NULL) OutOfBoundsException;
%varargs (void* null=NULL) TinyXMLError;
%varargs (void* null=NULL) DifferentModel;
%varargs (void* null=NULL) UnsupportedFileType;
%varargs (void* null=NULL) UnknownInternalError;
%varargs (void* null=NULL) ArrayMessageWriteConflict;
%varargs (void* null=NULL) VisualisationException;

%exceptionclass FGPUException;

%include "flamegpu/exception/FGPUException.h"
// Generic exception handling
%include "exception.i"
%exception {
    try {
        $action
    }
    catch (UnsupportedFileType& e) {
        FGPUException *ecopy = new UnsupportedFileType(e);
        PyObject *err = SWIG_NewPointerObj(ecopy, SWIGTYPE_p_UnsupportedFileType, 1);
        PyErr_SetObject(SWIG_Python_ExceptionType(SWIGTYPE_p_UnsupportedFileType), err);
        SWIG_fail;
    } 
    catch (...) {
        SWIG_exception(SWIG_RuntimeError, "Unknown Exception");
    } 
}

/* Include Simulation and CUDAModel */
%feature("flatnested");     // flat nested on to ensure Config is included
%rename (CUDAAgentModel_Config) CUDAAgentModel::Config;
%rename (Simulation_Config) Simulation::Config;
%include <argcargv.i>                                           // Include and apply to swig library healer for processing argc and argv values
%apply (int ARGC, char **ARGV) { (int argc, const char **) }    // This is required for CUDAAgentModel.initialise() 
%include "flamegpu/sim/Simulation.h"
%include "flamegpu/gpu/CUDAAgentModel.h"
%feature("flatnested", ""); // flat nested off

//%include "flamegpu/runtime/flamegpu_device_api.h"
%ignore VarOffsetStruct; // not required but defined in flamegpu_host_new_agent_api
%include "flamegpu/runtime/flamegpu_host_api.h"
%include "flamegpu/runtime/flamegpu_host_new_agent_api.h"
%include "flamegpu/runtime/flamegpu_host_agent_api.h"
%include "flamegpu/runtime/utility/HostRandom.cuh"
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
%feature("flatnested", "");     // flat nested off


/* Instanciate template versions of agent functions from the API */
TEMPLATE_VARIABLE_INSTANTIATE_N(newVariable, AgentDescription::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE_N(setVariable, AgentInstance::setVariable)
TEMPLATE_VARIABLE_INSTANTIATE(getVariable, AgentInstance::getVariable)

/* Instanciate template versions of new and get message types from the API */
%template(newMessageBruteForce) ModelDescription::newMessage<MsgBruteForce>;
%template(newMessageSpatial2D) ModelDescription::newMessage<MsgSpatial2D>;
%template(newMessageSpatial3D) ModelDescription::newMessage<MsgSpatial3D>;
%template(newMessageArray) ModelDescription::newMessage<MsgArray>;
%template(newMessageArray2D) ModelDescription::newMessage<MsgArray2D>;
%template(newMessageArray3D) ModelDescription::newMessage<MsgArray3D>;

%template(getMessageBruteForce) ModelDescription::getMessage<MsgBruteForce>;
%template(getMessageSpatial2D) ModelDescription::getMessage<MsgSpatial2D>;
%template(getMessageSpatial3D) ModelDescription::getMessage<MsgSpatial3D>;
%template(getMessageArray) ModelDescription::getMessage<MsgArray>;
%template(getMessageArray2D) ModelDescription::getMessage<MsgArray2D>;
%template(getMessageArray3D) ModelDescription::getMessage<MsgArray3D>;

/* Instanciate template versions of message functions from the API */
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgBruteForce::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgSpatial2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgSpatial3D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray3D::Description::newVariable)

/* Instanciate template versions of host agent functions from the API */

/* Instanciate template versions of host environment functions from the API */




