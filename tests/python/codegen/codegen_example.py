"""
    Complex agent function to be tested with #line directive to ensure directives are placed as expected.
    IMPORTANT: The `test_codegen_integration.py::test_gpu_codegen_complex` tests are sensitive to line changes. 
    Any changes to the lines in this file may result in errors and as such the corresponding C++ should be checked validated.
"""

import pyflamegpu

@pyflamegpu.device_function
def an_example_device_function(a : int) -> int:
    return a + 1

@pyflamegpu.agent_function
def agent_func_complex_example(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone):
    var_int1 = pyflamegpu.getVariableInt("var_int1")
    var_int2 = pyflamegpu.getVariableInt("var_int2")
    var_float1 = pyflamegpu.getVariableFloat("var_float1")
    var_float2 = pyflamegpu.getVariableFloat("var_float2")
    env_int1 = pyflamegpu.environment.getPropertyInt("env_int1")
    # condition
    if (var_int1 == var_int2):
        # assignment
        var_float1 = 10
    # assignment with expression
    var_float2 = var_float1*var_float1*0.01
    # loop
    for i in range(10):
        # assignment increment
        var_int1 += 1
    # message loop with assignment
    i = 0
    for m in message_in:
        i += m.getIndex()
    # python like if else inline
    var_int2 = 0 if (env_int1 == 0) else int(i)
    # set variable
    pyflamegpu.setVariableInt("var_int1", var_int1)
    # return
    return pyflamegpu.ALIVE

###################################################################################################
# Changes after this point are not subject to line placement sensitivity within integration tests #
###################################################################################################

# A stringified example of the above agent function and device function
agent_func_complex_example_str = """
@pyflamegpu.device_function
def an_example_device_function(a : int) -> int:
    return a + 1

@pyflamegpu.agent_function
def agent_func_complex_example(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone):
    var_int1 = pyflamegpu.getVariableInt("var_int1")
    var_int2 = pyflamegpu.getVariableInt("var_int2")
    var_float1 = pyflamegpu.getVariableFloat("var_float1")
    var_float2 = pyflamegpu.getVariableFloat("var_float2")
    env_int1 = pyflamegpu.environment.getPropertyInt("env_int1")
    # condition
    if (var_int1 == var_int2):
        # assignment
        var_float1 = 10
    # assignment with expression
    var_float2 = var_float1*var_float1*0.01
    # loop
    for i in range(10):
        # assignment increment
        var_int1 += 1
    # message loop with assignment
    i = 0
    for m in message_in:
        i += m.getIndex()
    # python like if else inline
    var_int2 = 0 if (env_int1 == 0) else int(i)
    # set variable
    pyflamegpu.setVariableInt("var_int1", var_int1)
    # return
    return pyflamegpu.ALIVE
"""


# Expected output when code generating the above agent function when passed as a callable
# Note linenumbers represent the actual line number in this source module 
agent_func_complex_example_file_cpp = """
#line 10 "codegen_example.py"
FLAMEGPU_DEVICE_FUNCTION int an_example_device_function(int a){
#line 11 "codegen_example.py"
    return (a + 1);
}

#line 14 "codegen_example.py"
FLAMEGPU_AGENT_FUNCTION(agent_func_complex_example, flamegpu::MessageBruteForce, flamegpu::MessageNone){
#line 15 "codegen_example.py"
    auto var_int1 = FLAMEGPU->getVariable<int>("var_int1");
#line 16 "codegen_example.py"
    auto var_int2 = FLAMEGPU->getVariable<int>("var_int2");
#line 17 "codegen_example.py"
    auto var_float1 = FLAMEGPU->getVariable<float>("var_float1");
#line 18 "codegen_example.py"
    auto var_float2 = FLAMEGPU->getVariable<float>("var_float2");
#line 19 "codegen_example.py"
    auto env_int1 = FLAMEGPU->environment.getProperty<int>("env_int1");
#line 21 "codegen_example.py"
    if (var_int1 == var_int2){
#line 23 "codegen_example.py"
        var_float1 = 10;
    }
#line 25 "codegen_example.py"
    var_float2 = ((var_float1 * var_float1) * 0.01);
#line 27 "codegen_example.py"
    for (int i=0;i<10;i++){
#line 29 "codegen_example.py"
        var_int1 += 1;
    }
#line 31 "codegen_example.py"
    auto i = 0;
#line 32 "codegen_example.py"
    for (const auto& m : FLAMEGPU->message_in){
#line 33 "codegen_example.py"
        i += m.getIndex();
    }
#line 35 "codegen_example.py"
    var_int2 = env_int1 == 0 ? 0 : static_cast<int>(i);
#line 37 "codegen_example.py"
    FLAMEGPU->setVariable<int>("var_int1", var_int1);
#line 39 "codegen_example.py"
    return flamegpu::ALIVE;
}
"""

agent_func_complex_example_dynamic_cpp = """
#line 1 "DynamicPython"
FLAMEGPU_DEVICE_FUNCTION int an_example_device_function(int a){
#line 1 "DynamicPython"
    return (a + 1);
}

#line 2 "DynamicPython"
FLAMEGPU_AGENT_FUNCTION(agent_func_complex_example, flamegpu::MessageBruteForce, flamegpu::MessageNone){
#line 3 "DynamicPython"
    auto var_int1 = FLAMEGPU->getVariable<int>("var_int1");
#line 4 "DynamicPython"
    auto var_int2 = FLAMEGPU->getVariable<int>("var_int2");
#line 5 "DynamicPython"
    auto var_float1 = FLAMEGPU->getVariable<float>("var_float1");
#line 6 "DynamicPython"
    auto var_float2 = FLAMEGPU->getVariable<float>("var_float2");
#line 7 "DynamicPython"
    auto env_int1 = FLAMEGPU->environment.getProperty<int>("env_int1");
#line 9 "DynamicPython"
    if (var_int1 == var_int2){
#line 11 "DynamicPython"
        var_float1 = 10;
    }
#line 13 "DynamicPython"
    var_float2 = ((var_float1 * var_float1) * 0.01);
#line 15 "DynamicPython"
    for (int i=0;i<10;i++){
#line 17 "DynamicPython"
        var_int1 += 1;
    }
#line 19 "DynamicPython"
    auto i = 0;
#line 20 "DynamicPython"
    for (const auto& m : FLAMEGPU->message_in){
#line 21 "DynamicPython"
        i += m.getIndex();
    }
#line 23 "DynamicPython"
    var_int2 = env_int1 == 0 ? 0 : static_cast<int>(i);
#line 25 "DynamicPython"
    FLAMEGPU->setVariable<int>("var_int1", var_int1);
#line 27 "DynamicPython"
    return flamegpu::ALIVE;
}"""

agent_func_complex_example_str_cpp = """
#line 3 "PythonString"
FLAMEGPU_DEVICE_FUNCTION int an_example_device_function(int a){
#line 4 "PythonString"
    return (a + 1);
}

#line 7 "PythonString"
FLAMEGPU_AGENT_FUNCTION(agent_func_complex_example, flamegpu::MessageBruteForce, flamegpu::MessageNone){
#line 8 "PythonString"
    auto var_int1 = FLAMEGPU->getVariable<int>("var_int1");
#line 9 "PythonString"
    auto var_int2 = FLAMEGPU->getVariable<int>("var_int2");
#line 10 "PythonString"
    auto var_float1 = FLAMEGPU->getVariable<float>("var_float1");
#line 11 "PythonString"
    auto var_float2 = FLAMEGPU->getVariable<float>("var_float2");
#line 12 "PythonString"
    auto env_int1 = FLAMEGPU->environment.getProperty<int>("env_int1");
#line 14 "PythonString"
    if (var_int1 == var_int2){
#line 16 "PythonString"
        var_float1 = 10;
    }
#line 18 "PythonString"
    var_float2 = ((var_float1 * var_float1) * 0.01);
#line 20 "PythonString"
    for (int i=0;i<10;i++){
#line 22 "PythonString"
        var_int1 += 1;
    }
#line 24 "PythonString"
    auto i = 0;
#line 25 "PythonString"
    for (const auto& m : FLAMEGPU->message_in){
#line 26 "PythonString"
        i += m.getIndex();
    }
#line 28 "PythonString"
    var_int2 = env_int1 == 0 ? 0 : static_cast<int>(i);
#line 30 "PythonString"
    FLAMEGPU->setVariable<int>("var_int1", var_int1);
#line 32 "PythonString"
    return flamegpu::ALIVE;
}"""