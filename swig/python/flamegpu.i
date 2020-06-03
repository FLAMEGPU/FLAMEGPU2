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

%module pyflamegpu



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

//#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
//#include "flamegpu/runtime/flamegpu_host_agent_api.h"
#include "flamegpu/runtime/messaging.h"


#include "flamegpu/runtime/AgentFunction_shim.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
%}

/* Disable non RTC function and function condtion set methods */
%ignore AgentDescription::newFunction;
%ignore AgentFunctionDescription::getFunctionPtr;
%ignore AgentFunctionDescription::setFunctionCondition;
%ignore AgentFunctionDescription::getConditionPtr;



/* SWIG header includes used to generate wrappers */
%include "flamegpu/model/ModelDescription.h"
%include "flamegpu/model/AgentDescription.h"
%include "flamegpu/model/AgentFunctionDescription.h"
//%include "flamegpu/model/EnvironmentDescription.h"
%include "flamegpu/model/LayerDescription.h"
%include "flamegpu/pop/AgentPopulation.h"
%include "flamegpu/pop/AgentInstance.h"

%include "flamegpu/gpu/CUDAAgentModel.h"

//%include "flamegpu/runtime/flamegpu_device_api.h"
//%include "flamegpu/runtime/flamegpu_host_api.h"
//#include "flamegpu/runtime/flamegpu_host_agent_api.h" // issues with cub and swig

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

/* Instanciate template versions of new message types from the API */
%template(newMessageBruteForce) ModelDescription::newMessage<MsgBruteForce>;
%template(newMessageSpatial2D) ModelDescription::newMessage<MsgSpatial2D>;
%template(newMessageSpatial3D) ModelDescription::newMessage<MsgSpatial3D>;
%template(newMessageArray) ModelDescription::newMessage<MsgArray>;
%template(newMessageArray2D) ModelDescription::newMessage<MsgArray2D>;
%template(newMessageArray3D) ModelDescription::newMessage<MsgArray3D>;

/* Instanciate template versions of message functions from the API */
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgBruteForce::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgSpatial2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgSpatial3D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray2D::Description::newVariable)
TEMPLATE_VARIABLE_INSTANTIATE(newVariable, MsgArray3D::Description::newVariable)
