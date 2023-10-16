import codecs
import os
import sys
import pytest
import unittest
import ast
import pyflamegpu.codegen
import astpretty


DEBUG_OUT = True
EXCEPTION_MSG_CHECKING = True

# Standard python syntax

py_for_else = """\
for x in range(10):
    break
else:
    y = 2
"""

py_for_range_arg1 = """\
for x in range(10):
    break
"""
cpp_for_range_arg1 = """\
for (int x=0;x<10;x++){
    break;
}
"""

py_for_range_arg2 = """\
for x in range(2, 11):
    break
"""
cpp_for_range_arg2 = """\
for (int x=2;x<11;x++){
    break;
}
"""

py_for_range_arg3 = """\
for x in range(3, 12, 4):
    break
"""
cpp_for_range_arg3 = """\
for (int x=3;x<12;x+=4){
    break;
}
"""

py_for_unsupported = """\
for x in something:
    break
"""

py_while_else = """\
while True:
    break
else:
    y = 2
"""

py_while = """\
while True:
    break
"""
cpp_while = """\
while (true){
    break;
}
"""

py_try = """\
try:
    1 / 0
except Exception as e:
    pass
"""

py_async_func = """\
async def async_function():
    pass
"""

py_class_decorator = """\
@f1(arg)
@f2
class Foo: pass
"""

py_elif1 = """\
if cond1:
    break
elif cond2:
    break
else:
    break
"""

cpp_elif1 = """\
if (cond1){
    break;
}
else if (cond2){
    break;
}
else{
    break;
}
"""

py_elif2 = """\
if cond1:
    break
elif cond2:
    break
"""

cpp_elif2 = """\
if (cond1){
    break;
}
else if (cond2){
    break;
}
"""

py_var_existing = """\
a = 1
a = 2
a += 3
"""
cpp_var_existing = """\
constexpr auto a = 1;
a = 2;
a += 3;
"""

py_var_scope = """\
for i in range(10):
    foo = pyflamegpu.getVariableInt("foo")
for i in range(10):
    foo = pyflamegpu.getVariableInt("foo")
"""
cpp_var_scope = """\
for (int i=0;i<10;i++){
    auto foo = FLAMEGPU->getVariable<int>("foo");
}
for (int i=0;i<10;i++){
    auto foo = FLAMEGPU->getVariable<int>("foo");
}
"""

py_with_simple = """\
with f():
    suite1
"""

py_with_as = """\
with f() as x:
    suite1
"""


py_async_function_def = """\
async def f():
    suite1
"""

py_async_for = """\
async for _ in reader:
    suite1
"""

py_async_with = """\
async with g():
    suite1
"""

py_async_with_as = """\
async with g() as x:
    suite1
"""

# FGPU functionality
py_fgpu_constexpr = """\
a = 12

@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    b = 13
"""

cpp_fgpu_constexpr = """\
constexpr auto a = 12;

FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto b = 13;
}
"""



py_fgpu_types = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    f = pyflamegpu.getVariableFloat("f")
    d = pyflamegpu.getVariableDouble("d")
    i = pyflamegpu.getVariableInt("i")
    ui = pyflamegpu.getVariableUInt("ui")
    i8 = pyflamegpu.getVariableInt8("i8")
    ui8 = pyflamegpu.getVariableUInt8("ui8")
    c = pyflamegpu.getVariableChar("c")
    uc = pyflamegpu.getVariableUChar("uc")
    i16 = pyflamegpu.getVariableInt16("i16")
    ui16 = pyflamegpu.getVariableUInt16("ui16")
    i32 = pyflamegpu.getVariableInt32("i32")
    ui32 = pyflamegpu.getVariableUInt32("ui32")
    i64 = pyflamegpu.getVariableInt64("i64")
    ui64 = pyflamegpu.getVariableUInt64("ui64")
"""

cpp_fgpu_types = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto f = FLAMEGPU->getVariable<float>("f");
    auto d = FLAMEGPU->getVariable<double>("d");
    auto i = FLAMEGPU->getVariable<int>("i");
    auto ui = FLAMEGPU->getVariable<unsigned int>("ui");
    auto i8 = FLAMEGPU->getVariable<int_8>("i8");
    auto ui8 = FLAMEGPU->getVariable<uint_8>("ui8");
    auto c = FLAMEGPU->getVariable<char>("c");
    auto uc = FLAMEGPU->getVariable<unsigned char>("uc");
    auto i16 = FLAMEGPU->getVariable<int_16>("i16");
    auto ui16 = FLAMEGPU->getVariable<uint_16>("ui16");
    auto i32 = FLAMEGPU->getVariable<int_32>("i32");
    auto ui32 = FLAMEGPU->getVariable<uint_32>("ui32");
    auto i64 = FLAMEGPU->getVariable<int_64>("i64");
    auto ui64 = FLAMEGPU->getVariable<uint_64>("ui64");
}
"""

py_fgpu_array_types = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    f = pyflamegpu.getVariableFloatArray1("f")
    d = pyflamegpu.getVariableDoubleArray2("d")
    i = pyflamegpu.getVariableIntArray3("i")
    ui = pyflamegpu.getVariableUIntArray4("ui")
    i8 = pyflamegpu.getVariableInt8Array5("i8")
    ui8 = pyflamegpu.getVariableUInt8Array6("ui8")
    c = pyflamegpu.getVariableCharArray7("c")
    uc = pyflamegpu.getVariableUCharArray8("uc")
    i16 = pyflamegpu.getVariableInt16Array9("i16")
    ui16 = pyflamegpu.getVariableUInt16Array10("ui16")
    i32 = pyflamegpu.getVariableInt32Array11("i32")
    ui32 = pyflamegpu.getVariableUInt32Array12("ui32")
    i64 = pyflamegpu.getVariableInt64Array13("i64")
    ui64 = pyflamegpu.getVariableUInt64Array14("ui64")
"""

cpp_fgpu_array_types = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto f = FLAMEGPU->getVariable<float, 1>("f");
    auto d = FLAMEGPU->getVariable<double, 2>("d");
    auto i = FLAMEGPU->getVariable<int, 3>("i");
    auto ui = FLAMEGPU->getVariable<unsigned int, 4>("ui");
    auto i8 = FLAMEGPU->getVariable<int_8, 5>("i8");
    auto ui8 = FLAMEGPU->getVariable<uint_8, 6>("ui8");
    auto c = FLAMEGPU->getVariable<char, 7>("c");
    auto uc = FLAMEGPU->getVariable<unsigned char, 8>("uc");
    auto i16 = FLAMEGPU->getVariable<int_16, 9>("i16");
    auto ui16 = FLAMEGPU->getVariable<uint_16, 10>("ui16");
    auto i32 = FLAMEGPU->getVariable<int_32, 11>("i32");
    auto ui32 = FLAMEGPU->getVariable<uint_32, 12>("ui32");
    auto i64 = FLAMEGPU->getVariable<int_64, 13>("i64");
    auto ui64 = FLAMEGPU->getVariable<uint_64, 14>("ui64");
}
"""

py_fgpu_unknown_type = """\
i = pyflamegpu.getVariableUnknownType("i")
"""

py_fgpu_for_msg_input = """\
for msg in message_in:
    f = msg.getVariableFloat("f")
    d = msg.getVariableDouble("d")
    i = msg.getVariableInt("i")
"""
cpp_fgpu_for_msg_input = """\
for (const auto& msg : FLAMEGPU->message_in){
    auto f = msg.getVariable<float>("f");
    auto d = msg.getVariable<double>("d");
    auto i = msg.getVariable<int>("i");
}
"""

py_fgpu_name_not_attr = """\
@pyflamegpu.agent_function
def movement_request(message_in: pyflamegpu.MessageArray2D, message_out: pyflamegpu.MessageArray2D):
  AGENT_START_COUNT = int(2)
  return pyflamegpu.ALIVE
"""
cpp_fgpu_name_not_attr = """\
FLAMEGPU_AGENT_FUNCTION(movement_request, flamegpu::MessageArray2D, flamegpu::MessageArray2D){
    auto AGENT_START_COUNT = static_cast<int>(2);
    return flamegpu::ALIVE;
}
"""

py_fgpu_for_msg_input_var = """\
for m in message_in:
    i = m.getVariableInt("i")
"""
cpp_fgpu_for_msg_input_var = """\
for (const auto& m : FLAMEGPU->message_in){
    auto i = m.getVariable<int>("i");
}
"""

py_fgpu_for_msg_input_wrap = """\
for m in message_in.wrap(x, y): 
    pass
"""
cpp_fgpu_for_msg_input_wrap = """\
for (const auto& m : FLAMEGPU->message_in.wrap(x, y)){
    ;
}
"""

py_fgpu_for_msg_input_unsupported = """\
for m in message_in.radius(): 
    pass
"""

py_fgpu_for_msg_input_funcs = """\
for m in message_in:
    i = m.getIndex()
"""
cpp_fgpu_for_msg_input_funcs = """\
for (const auto& m : FLAMEGPU->message_in){
    auto i = m.getIndex();
}
"""

py_fgpu_for_msg_input_func_unknown = """\
for m in message_in:
    i = m.unsupported()
"""

py_fgpu_for_msg_input_math_func = """\
for m in message_in:
    i = math.sqrt()
"""
cpp_fgpu_for_msg_input_math_func = """\
for (const auto& m : FLAMEGPU->message_in){
    auto i = sqrt();
}
"""

py_fgpu_standalone_msg_input = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    m = message_in.at(1)
    pass
    n = m.getVariableInt("foo")
"""
cpp_fgpu_standalone_msg_input = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto m = FLAMEGPU->message_in.at(1);
    ;
    auto n = m.getVariable<int>("foo");
}
"""

py_fgpu_for_msg_input_args = """\
for m in message_in(x, y, z) : 
    pass
"""
cpp_fgpu_for_msg_input_args = """\
for (const auto& m : FLAMEGPU->message_in(x, y, z)){
    ;
}
"""

py_fgpu_msg_output = """\
message_out.setVariableFloat("f", f)
message_out.setVariableDouble("d", d)
message_out.setVariableInt("i", i)
"""
cpp_fgpu_msg_output = """\
FLAMEGPU->message_out.setVariable<float>("f", f);
FLAMEGPU->message_out.setVariable<double>("d", d);
FLAMEGPU->message_out.setVariable<int>("i", i);
"""

py_fgpu_graph_fns = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    fgraph = pyflamegpu.environment.getDirectedGraph("fgraph")

    # Fetch the ID of the vertex at index 0
    vertex_id = fgraph.getVertexID(0)
    # Fetch the index of the vertex with ID 1
    vertex_index = fgraph.getVertexIndex(1)

    # Access a property of vertex with ID 1
    bar_0 = fgraph.getVertexPropertyFloatArray2("bar", 0)

    # Fetch the source and destination indexes from the edge at index 0
    source_index = fgraph.getEdgeSource(0)
    destination_index = fgraph.getEdgeDestination(0)

    # Fetch the index of the edge from vertex ID 1 to vertex ID 2
    edge_index = fgraph.getEdgeIndex(1, 2)

    # Access a property of edge with source ID 1, destination ID 2
    foo2 = fgraph.getEdgePropertyInt("foo", edge_index);
"""
cpp_fgpu_graph_fns = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto fgraph = FLAMEGPU->environment.getDirectedGraph("fgraph");
    auto vertex_id = fgraph.getVertexID(0);
    auto vertex_index = fgraph.getVertexIndex(1);
    auto bar_0 = fgraph.getVertexProperty<float, 2>("bar", 0);
    auto source_index = fgraph.getEdgeSource(0);
    auto destination_index = fgraph.getEdgeDestination(0);
    auto edge_index = fgraph.getEdgeIndex(1, 2);
    auto foo2 = fgraph.getEdgeProperty<int>("foo", edge_index);
}
"""

py_fgpu_for_graph_in_fns = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    # Iterate the edges joining the vertex with ID 1
    fgraph = pyflamegpu.environment.getDirectedGraph("fgraph")
    for edge in fgraph.inEdges(vertex_index):
        # Read the current edges' source vertex index
        src_vertex_index = edge.getEdgeSource()
        # Read a property from the edge
        foo = edge.getPropertyInt("foo")
        bar = edge.getPropertyFloatArray2("bar", 0)
"""
cpp_fgpu_for_graph_in_fns = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto fgraph = FLAMEGPU->environment.getDirectedGraph("fgraph");
    for (const auto& edge : fgraph.inEdges(vertex_index)){
        auto src_vertex_index = edge.getEdgeSource();
        auto foo = edge.getProperty<int>("foo");
        auto bar = edge.getProperty<float, 2>("bar", 0);
    }
}
"""

py_fgpu_for_graph_out_fns = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    # Iterate the edges leaving the vertex with ID 1
    fgraph = pyflamegpu.environment.getDirectedGraph("fgraph")
    for edge in fgraph.outEdges(vertex_index):
        # Read the current edges' destination vertex index
        dest_vertex_index = edge.getEdgeDestination()
        # Read a property from the edge
        foo = edge.getPropertyInt("foo")
        bar = edge.getPropertyFloatArray2("bar", 0)
"""
cpp_fgpu_for_graph_out_fns = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto fgraph = FLAMEGPU->environment.getDirectedGraph("fgraph");
    for (const auto& edge : fgraph.outEdges(vertex_index)){
        auto dest_vertex_index = edge.getEdgeDestination();
        auto foo = edge.getProperty<int>("foo");
        auto bar = edge.getProperty<float, 2>("bar", 0);
    }
}
"""


py_fgpu_macro_env_permitted = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    a = pyflamegpu.environment.getMacroPropertyInt('a') 
    a += 1
    a.exchange(b) 
    a.CAS(b, c)
    a.min(b)
    a.max(b)
"""
cpp_fgpu_macro_env_permitted = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto a = FLAMEGPU->environment.getMacroProperty<int>("a");
    a += 1;
    a.exchange(b);
    a.CAS(b, c);
    a.min(b);
    a.max(b);
}
"""

py_fgpu_macro_env_function = """\
a = pyflamegpu.environment.getMacroPropertyInt('a')"
a += 1
a.exchange(b)
a.CAS(b, c)
a.min(b)
a.max(b)
"""

py_fgpu_agent_func = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    pass
"""
py_fgpu_agent_func_docstring = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    \"\"\"
    This is a docstring
    \"\"\"
    pass
"""
py_fgpu_agent_func_comment = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    # This is a comment
    pass
"""

cpp_fgpu_agent_func = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageBruteForce){
    ;
}
"""

py_fgpu_agent_func_no_input_msg_type = """\
@pyflamegpu.agent_function
def func(message_in, message_out: pyflamegpu.MessageBruteForce):
    pass
"""

py_fgpu_agent_func_no_output_msg_type = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out):
    pass
"""

py_fgpu_agent_func_extra_args = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageBruteForce, other):
    pass
"""

py_fgpu_agent_func_extra_args = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageBruteForce, other):
    pass
"""

py_fgpu_agent_func_return_type = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce) -> int :
    pass
"""
cpp_fgpu_agent_func_return_type = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageBruteForce){
    ;
}
"""

py_fgpu_agent_func_check_agent_name_state = """\
@pyflamegpu.agent_function
def func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone) :
    a = pyflamegpu.isAgent("foo")
    b = pyflamegpu.isState("bar");
"""
cpp_fgpu_agent_func_check_agent_name_state = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageNone){
    auto a = FLAMEGPU->isAgent("foo");
    auto b = FLAMEGPU->isState("bar");
}
"""

py_fgpu_device_func_args = """\
@pyflamegpu.device_function
def func(x: int) -> int :
    pass
"""

cpp_fgpu_device_func_args = """\
FLAMEGPU_DEVICE_FUNCTION int func(int x){
    ;
}
"""

py_fgpu_device_func_no_type = """\
@pyflamegpu.device_function
def func(x) -> int :
    pass
"""

py_fgpu_device_func_no_return_type = """\
@pyflamegpu.device_function
def func(x: int) :
    pass
"""
cpp_fgpu_device_func_no_return_type = """\
FLAMEGPU_DEVICE_FUNCTION void func(int x){
    ;
}
"""

py_fgpu_device_func_no_decorator = """\
def func(x: int) -> int :
    pass
"""

py_fgpu_device_func_arg_modified = """\
@pyflamegpu.device_function
def func(x: int) :
    x = 10
"""
cpp_fgpu_device_func_arg_modified = """\
FLAMEGPU_DEVICE_FUNCTION void func(int x){
    x = 10;
}
"""

py_fgpu_cond_func = """\
@pyflamegpu.agent_function_condition
def conditionalFunc() -> bool :
    return True
"""

cpp_fgpu_cond_func = """\
FLAMEGPU_AGENT_FUNCTION_CONDITION(conditionalFunc){
    return true;
}
"""

py_fgpu_cond_func_args = """\
@pyflamegpu.agent_function_condition
def conditionalFunc(arg: int) -> bool:
    return True
"""
py_fgpu_cond_func_no_return_type = """\
@pyflamegpu.agent_function_condition
def conditionalFunc() :
    return True
"""
py_fgpu_cond_func_wrong_return_type = """\
@pyflamegpu.agent_function_condition
def conditionalFunc() -> int :
    return True
"""


py_fgpu_device_local_args_stack = """\
@pyflamegpu.device_function
def funcA(x: int) :
    a = 1
    x = 10

@pyflamegpu.device_function
def funcB(y: int) :
    a = 1
    y= 10
"""
cpp_fgpu_device_local_args_stack = """\
FLAMEGPU_DEVICE_FUNCTION void funcA(int x){
    auto a = 1;
    x = 10;
}

FLAMEGPU_DEVICE_FUNCTION void funcB(int y){
    auto a = 1;
    y = 10;
}
"""


class CodeGenTest(unittest.TestCase):


    def _checkExpected(self, source, expected):
        source = source.strip()
        tree = ast.parse(source)
        if DEBUG_OUT:
            astpretty.pprint(tree)
        code = pyflamegpu.codegen.codegen(tree)
        # remove new lines
        code = code.strip()
        expected = expected.strip()
        if DEBUG_OUT:
            print(f"Expected: {expected}")
            print(f"Output  : {code}")
        assert expected == code
            
    def _checkWarning(self, source, expected, warning_str):
        with pytest.warns() as record:
            self._checkExpected(source, expected)
        assert warning_str in str(record[0].message)
        
    def _checkException(self, source, exception_str):
        with pytest.raises(pyflamegpu.codegen.CodeGenException) as e:
            tree = ast.parse(source.strip())
            # code generate
            code = pyflamegpu.codegen.codegen(tree)
        if EXCEPTION_MSG_CHECKING:
            assert exception_str in str(e.value)
           
        

    def test_del_statement(self):
        self._checkException("del x, y, z", "Deletion not supported")

    def test_shifts(self):
        self._checkExpected("45 << 2", "(45 << 2);")
        self._checkExpected("13 >> 7", "(13 >> 7);")

    def test_for_else(self):
        self._checkException(py_for_else, "For else not supported")
        
    def test_for_range(self):
        # use of range based for loops (calling range with different number of arguments)
        self._checkExpected(py_for_range_arg1, cpp_for_range_arg1)
        self._checkExpected(py_for_range_arg2, cpp_for_range_arg2)
        self._checkExpected(py_for_range_arg3, cpp_for_range_arg3)   
        # check that non range function loops are rejected
        self._checkException(py_for_unsupported, "Range based for loops only support message iteration using 'message_in' or directed graph iterator")
       
    def test_while_else(self):
        self._checkException(py_while_else, "While else not supported")
        
    def test_while(self):
        self._checkExpected(py_while, cpp_while)

    def test_unary_parens(self):
        self._checkExpected("(-1)**7", "pow((-1), 7);")
        self._checkExpected("-1.**8", "(-pow(1.0, 8));")
        self._checkExpected("not True or False", "((!true) || false);")
        self._checkExpected("True or not False", "(true || (!false));")

    def test_integer_parens(self):
        self._checkException("3 .__abs__()", "Unsupported") # should resolve to unsupported function call syntax

    def test_huge_float(self):
        self._checkExpected("1e1000", "inf;")

    def test_min_int30(self):
        self._checkExpected(str(-2**31), "(-2147483648);")
        self._checkExpected(str(-2**63), "(-9223372036854775808);")

    def test_negative_zero(self):
        self._checkExpected("-0", "(-0);")
        self._checkExpected("-(0)", "(-0);")
        self._checkExpected("-0b0", "(-0);")
        self._checkExpected("-(0b0)", "(-0);")
        self._checkExpected("-0o0", "(-0);")
        self._checkExpected("-(0o0)", "(-0);")
        self._checkExpected("-0x0", "(-0);")
        self._checkExpected("-(0x0)", "(-0);")

    def test_lambda_parentheses(self):
        self._checkException("(lambda: int)()", "Lambda is not supported")

    def test_chained_comparisons(self):
        self._checkExpected("1 < 4 <= 5", "1 < 4 <= 5;")
        self._checkExpected("a is b is c is not d", "a == b == c != d;")

    def test_function_arguments(self):
        # only flame gpu functions or device functions are supported
        self._checkException("def f(): pass", "Function definitions require a")

    def test_relative_import(self):
        self._checkException("from . import fred", "Importing of modules not supported")

    def test_import_many(self):
        self._checkException("import fred, other", "Importing of modules not supported")

    def test_nonlocal(self):
        self._checkException("nonlocal x", "Use of 'nonlocal' not supported")

    def test_exceptions(self):
        self._checkException("raise Error", "Exception raising not supported")
        self._checkException(py_try, "Exceptions not supported")

    def test_bytes(self):
        self._checkException("b'123'", "Byte strings and Bytes function not supported")

    def test_strings(self):
        self._checkException('f"{value}"', "not supported")

    def test_set_literal(self):
        self._checkException("{'a', 'b', 'c'}", "Sets not supported")

    def test_comprehension(self):
        self._checkException("{x for x in range(5)}", "Set comprehension not supported")
        self._checkException("{x: x*x for x in range(10)}", "Dictionary comprehension not supported")

    def test_dict_with_unpacking(self):
        self._checkException("{**x}", "Dictionaries not supported")
        self._checkException("{a: b, **x}", "Dictionaries not supported")

    def test_async_comp_and_gen_in_async_function(self):
        self._checkException(py_async_func, "Async functions not supported")

    def test_async_comprehension(self):
        self._checkException("{i async for i in aiter() if i % 2}", "Set comprehension not supported")
        self._checkException("[i async for i in aiter() if i % 2]", "List comprehension not supported")
        self._checkException("{i: -i async for i in aiter() if i % 2}", "Dictionary comprehension not supported")

    def test_async_generator_expression(self):
        self._checkException("(i ** 2 async for i in agen())", "Generator expressions not supported")
        self._checkException("(i - 1 async for i in agen() if i % 2)", "Generator expressions not supported")

    def test_class(self):
        self._checkException("class Foo: pass", "Class definitions not supported")
        self._checkException(py_class_decorator, "Class definitions not supported")


    def test_elifs(self):
        self._checkExpected(py_elif1, cpp_elif1)
        self._checkExpected(py_elif2, cpp_elif2)


    def test_starred_assignment(self):
        self._checkException("a, *b = seq", "Assignment to complex expressions not supported")

    def test_variable_annotation(self):
        self._checkExpected("a: int", "int a;")
        self._checkExpected("a: int = 0", "int a = 0;")
        self._checkExpected("a: int = None", "int a = 0;")
        self._checkException("some_list: List[int]", "Not a supported type")
        self._checkException("some_list: List[int] = []", "Not a supported type")
        self._checkException("t: Tuple[int, ...] = (1, 2, 3)", "Not a supported type")

    def test_variable_existing(self):
        self._checkExpected(py_var_existing, cpp_var_existing)
        
    def test_variable_scope(self):
        self._checkExpected(py_var_scope, cpp_var_scope)


    def test_with(self):
        self._checkException(py_with_simple, "With not supported")
        self._checkException(py_with_as, "With not supported")
        self._checkException(py_async_with, "Async with not supported")
        self._checkException(py_async_with_as, "Async with not supported")

    def test_async_function_def(self):
        self._checkException(py_async_function_def, "Async functions not supported")

    def test_async_for(self):
        self._checkException(py_async_for, "Async for not supported")

    def test_check_name_not_attr(self):
        # This case previously caused codegen to throw an exception (Bug #1141)
        # I'm not sure this is still strictly correct, as the python cast is not being changed to C style
        self._checkExpected(py_fgpu_name_not_attr, cpp_fgpu_name_not_attr)

# FLAME GPU specific functionality

    # numpy types
    

    def test_fgpu_supported_types(self):
        self._checkExpected("a: numpy.byte", "char a;")
        self._checkExpected("a: numpy.ubyte", "unsigned char a;"),
        self._checkExpected("a: numpy.short", "short a;")
        self._checkExpected("a: numpy.ushort", "unsigned short a;")
        self._checkExpected("a: numpy.intc", "int a;")
        self._checkExpected("a: numpy.uintc", "unsigned int a;")
        self._checkExpected("a: numpy.uint", "unisgned int a;")
        self._checkExpected("a: numpy.longlong", "long long a;")
        self._checkExpected("a: numpy.ulonglong", "unsigned long long a;")
        self._checkExpected("a: numpy.half", "half a;")
        self._checkExpected("a: numpy.single", "float a;")
        self._checkExpected("a: numpy.double", "double a;")
        self._checkExpected("a: numpy.longdouble", "long double a;")
        self._checkExpected("a: numpy.bool_", "bool a;")
        self._checkExpected("a: numpy.bool8", "bool a;")
        # sized aliases
        self._checkExpected("a: numpy.int_", "long a;")
        self._checkExpected("a: numpy.int8", "int8_t a;"),
        self._checkExpected("a: numpy.int16", "int16_t a;")
        self._checkExpected("a: numpy.int32", "int32_t a;")
        self._checkExpected("a: numpy.int64", "int64_t a;")
        self._checkExpected("a: numpy.intp", "intptr_t a;")
        self._checkExpected("a: numpy.uint_", "long a;")
        self._checkExpected("a: numpy.uint8", "uint8_t a;")
        self._checkExpected("a: numpy.uint16", "uint16_t a;")
        self._checkExpected("a: numpy.uint32", "uint32_t a;")
        self._checkExpected("a: numpy.uint64", "uint64_t a;")
        self._checkExpected("a: numpy.uintp", "uintptr_t a;")
        self._checkExpected("a: numpy.float_", "float a;")
        self._checkExpected("a: numpy.float16", "half a;")
        self._checkExpected("a: numpy.float32", "float a;")
        self._checkExpected("a: numpy.float64", "double a;")
        # check unsupported
        self._checkException("a: numpy.unsupported", "numpy.unsupported is not a supported numpy type")
        
    
    def test_fgpu_constexpr(self):
        self._checkExpected(py_fgpu_constexpr, cpp_fgpu_constexpr)
    
    # getVariable and types
    
    def test_fgpu_types(self):
        # Check the type translation for getVariable function of FLAMEGPU singleton
        self._checkExpected(py_fgpu_types, cpp_fgpu_types)
        # Check the type translation for array type with getVariable function of FLAMEGPU singleton
        self._checkExpected(py_fgpu_array_types, cpp_fgpu_array_types)
        # Check unknown type
        self._checkException(py_fgpu_unknown_type, "is not a valid FLAME GPU type")       
        
    # message input
    
    def test_fgpu_for_msg_input(self):
        # Test message input
        self._checkExpected(py_fgpu_for_msg_input, cpp_fgpu_for_msg_input)
        # Test the use of a different message input variable within the for loop
        self._checkExpected(py_fgpu_for_msg_input_var, cpp_fgpu_for_msg_input_var)
        # Test message input iterator funcs
        self._checkExpected(py_fgpu_for_msg_input_funcs, cpp_fgpu_for_msg_input_funcs)    
        # Test message input with unknown function
        self._checkException(py_fgpu_for_msg_input_func_unknown, "Function 'unsupported' does not exist") 
        # Test math function inside message loop (Previously bug #1077)
        self._checkExpected(py_fgpu_for_msg_input_math_func, cpp_fgpu_for_msg_input_math_func) 
        # Test message input where message input requires arguments (e.g. spatial messaging)
        self._checkExpected(py_fgpu_for_msg_input_args, cpp_fgpu_for_msg_input_args)
        # Test to ensure that arguments are processed as local variables 
        self._checkExpected(py_fgpu_device_func_arg_modified, cpp_fgpu_device_func_arg_modified)
        # Test local variables of device functions to ensure locals are in fact local (by correctly specifying auto where required)
        self._checkExpected(py_fgpu_device_local_args_stack, cpp_fgpu_device_local_args_stack)
        # Test to ensure use of wrap is ok
        self._checkExpected(py_fgpu_for_msg_input_wrap, cpp_fgpu_for_msg_input_wrap)
        # Test to ensure 'message_in' does not allow non iterator functions
        self._checkException(py_fgpu_for_msg_input_unsupported, "Message input loop iterator 'radius' is not supported") # not currently raising an exception
    
    def test_fgpu_msg_input_standalone(self):
        # Test standalone input message (Previously bug #1110)
        self._checkExpected(py_fgpu_standalone_msg_input, cpp_fgpu_standalone_msg_input) 

    
    # message output
    
    def test_fgpu_msg_output(self):
        # Test message output 
        self._checkExpected(py_fgpu_msg_output, cpp_fgpu_msg_output)
        # Test message output with unknown function
        self._checkException("message_out.unsupported()", "Function 'unsupported' does not exist") 
        
    def test_fgpu_graph(self):
        self._checkExpected(py_fgpu_graph_fns, cpp_fgpu_graph_fns)
        self._checkExpected(py_fgpu_for_graph_in_fns, cpp_fgpu_for_graph_in_fns)
        self._checkExpected(py_fgpu_for_graph_out_fns, cpp_fgpu_for_graph_out_fns)
        
    # random
    
    def test_fgpu_random(self):
        self._checkExpected("pyflamegpu.random.uniformFloat()", "FLAMEGPU->random.uniform<float>();")
        self._checkExpected("pyflamegpu.random.uniformDouble()", "FLAMEGPU->random.uniform<double>();")
        self._checkExpected("pyflamegpu.random.uniformInt()", "FLAMEGPU->random.uniform<int>();")
        self._checkException("pyflamegpu.random.uniformBadType()", "'BadType' is not a valid FLAME GPU type")
        self._checkException("pyflamegpu.random.unsupported()", "Function 'unsupported' does not exist in pyflamegpu.random")
    
    # environment    
    def test_fgpu_environment(self):
        self._checkExpected("pyflamegpu.environment.getPropertyFloat('f')", 'FLAMEGPU->environment.getProperty<float>("f");')
        self._checkExpected("pyflamegpu.environment.getPropertyDouble('d')", 'FLAMEGPU->environment.getProperty<double>("d");')
        self._checkExpected("pyflamegpu.environment.getPropertyInt('i')", 'FLAMEGPU->environment.getProperty<int>("i");')
        self._checkExpected("pyflamegpu.environment.containsProperty('p')", 'FLAMEGPU->environment.containsProperty("p");')
        self._checkException("pyflamegpu.environment.getPropertyBadType()", "'BadType' is not a valid FLAME GPU type")
        self._checkException("pyflamegpu.environment.unsupported()", "Function 'unsupported' does not exist in pyflamegpu.environment")

    # macro environment
    def test_fgpu_macro_environment(self):
        # return a device macro property (not an array)
        self._checkExpected("pyflamegpu.environment.getMacroPropertyFloat('a')", 'FLAMEGPU->environment.getMacroProperty<float>("a");')
        # return a device macro property (array)
        self._checkExpected("pyflamegpu.environment.getMacroPropertyFloat('big_prop', 1, 2, 3)", 'FLAMEGPU->environment.getMacroProperty<float, 1, 2, 3>("big_prop");')
        # check for contains macro property
        self._checkExpected("pyflamegpu.environment.containsMacroProperty('p')", 'FLAMEGPU->environment.containsMacroProperty("p");')
        # perform a permitted function on a device macro property (inline)
        self._checkExpected("pyflamegpu.environment.getMacroPropertyInt('a').exchange(10)", 'FLAMEGPU->environment.getMacroProperty<int>("a").exchange(10);')
        # perform permitted operation on a macro device property variable
        self._checkExpected(py_fgpu_macro_env_permitted, cpp_fgpu_macro_env_permitted)
        # Possible additional test would be to do not allow a non device macro property to use the device macro functions (requires type checking)
        # Possible additional test could be to check that a macro device property does not call a function unsupported by the type (this also requires type checking and is probably overkill as it will be caught in compilation anyway)

    
    # agent functions, args renaming, arg types
    
    def test_fgpu_agent_func(self):
        self._checkExpected(py_fgpu_agent_func, cpp_fgpu_agent_func)

    def test_fgpu_agent_func_comments(self):
        """
        Comments and doc string should be removed and not translated
        """
        self._checkExpected(py_fgpu_agent_func_comment, cpp_fgpu_agent_func)
        self._checkExpected(py_fgpu_agent_func_docstring, cpp_fgpu_agent_func)
        
    def test_fgpu_agent_func_input_types(self):
        """ Try all the message input types by using a string replacement """
        # try all correct types
        for msg_type in pyflamegpu.codegen.CodeGenerator.fgpu_message_types:
            py_func = py_fgpu_agent_func.replace("pyflamegpu.MessageNone", msg_type)
            cpp_msg_type = msg_type.replace("pyflamegpu.", "flamegpu::")
            cpp_output = cpp_fgpu_agent_func.replace("flamegpu::MessageNone", cpp_msg_type)
            self._checkExpected(py_func, cpp_output)
        # try an incorrect type
        py_func = py_fgpu_agent_func.replace("pyflamegpu.MessageNone", "pyflamegpu.UnsupportedMessageType")
        self._checkException(py_func, "Message input type annotation not a supported message type")
        
    def test_fgpu_agent_func_output_types(self):
        """ Try all the message output types by using a string replacement """
        # try all correct types
        for msg_type in pyflamegpu.codegen.CodeGenerator.fgpu_message_types:
            py_func = py_fgpu_agent_func.replace("pyflamegpu.MessageBruteForce", msg_type)
            cpp_msg_type = msg_type.replace("pyflamegpu.", "flamegpu::")
            cpp_output = cpp_fgpu_agent_func.replace("flamegpu::MessageBruteForce", cpp_msg_type)
            self._checkExpected(py_func, cpp_output)
        # try an incorrect type
        py_func = py_fgpu_agent_func.replace("pyflamegpu.MessageBruteForce", "pyflamegpu.UnsupportedMessageType")
        self._checkException(py_func, "Message output type annotation not a supported message type")
        
    def test_fgpu_agent_func_incorrect_args(self):
        # Missing input message type
        self._checkException(py_fgpu_agent_func_no_input_msg_type, "Message input requires a supported type annotation")
        # Missing output message type
        self._checkException(py_fgpu_agent_func_no_output_msg_type, "Message output requires a supported type annotation")
        # Extra argument
        self._checkException(py_fgpu_agent_func_extra_args, "Expected two FLAME GPU function arguments")
        
    def test_fgpu_agent_func_return_type(self):
        """ Return type on an agent function raises a warning not error """
        self._checkWarning(py_fgpu_agent_func_return_type, cpp_fgpu_agent_func_return_type, "Function definition return type not supported")
        
    def test_fgpu_agent_func_check_agent_name_state(self):    
        self._checkExpected(py_fgpu_agent_func_check_agent_name_state, cpp_fgpu_agent_func_check_agent_name_state)
    
    # device functions, arg types and calling
    def test_fgpu_agent_func_condition(self):
        # check correct format
        self._checkExpected(py_fgpu_cond_func, cpp_fgpu_cond_func)
        # function is not allowed arguments but only a warning
        self._checkWarning(py_fgpu_cond_func_args, cpp_fgpu_cond_func, "Agent function conditions does not support arguments")
        # must have bool return type
        self._checkException(py_fgpu_cond_func_no_return_type, "Agent function conditions return type must be 'bool'")
        # only bool return type
        self._checkException(py_fgpu_cond_func_wrong_return_type, "Agent function conditions return type must be 'bool'")

    def test_fgpu_device_func(self):
        self._checkExpected(py_fgpu_device_func_args, cpp_fgpu_device_func_args)
        # function argument requires a type
        self._checkException(py_fgpu_device_func_no_type, "Device function argument requires type annotation")
        # no function return type should use void
        self._checkExpected(py_fgpu_device_func_no_return_type, cpp_fgpu_device_func_no_return_type)
        # function requires decorator
        self._checkException(py_fgpu_device_func_no_decorator, "Function definitions require a single pyflamegpu decorator of")
        

    def test_alive_dead(self):
        self._checkExpected(str("pyflamegpu.ALIVE"), "flamegpu::ALIVE;")
        self._checkExpected(str("pyflamegpu.DEAD"), "flamegpu::DEAD;")
    