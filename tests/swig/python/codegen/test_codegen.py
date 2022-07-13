import codecs
import os
import sys
import six
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
auto a = 1;
a = 2;
a += 3;
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




py_fgpu_types = """\
f = FLAMEGPU.getVariableFloat("f")
d = FLAMEGPU.getVariableDouble("d")
i = FLAMEGPU.getVariableInt("i")
ui = FLAMEGPU.getVariableUInt("ui")
i8 = FLAMEGPU.getVariableInt8("i8")
ui8 = FLAMEGPU.getVariableUInt8("ui8")
c = FLAMEGPU.getVariableChar("c")
uc = FLAMEGPU.getVariableUChar("uc")
i16 = FLAMEGPU.getVariableInt16("i16")
ui16 = FLAMEGPU.getVariableUInt16("ui16")
i32 = FLAMEGPU.getVariableInt32("i32")
ui32 = FLAMEGPU.getVariableUInt32("ui32")
i64 = FLAMEGPU.getVariableInt64("i64")
ui64 = FLAMEGPU.getVariableUInt64("ui64")
"""

cpp_fgpu_types = """\
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
"""

py_fgpu_array_types = """\
f = FLAMEGPU.getVariableFloatArray1("f")
d = FLAMEGPU.getVariableDoubleArray2("d")
i = FLAMEGPU.getVariableIntArray3("i")
ui = FLAMEGPU.getVariableUIntArray4("ui")
i8 = FLAMEGPU.getVariableInt8Array5("i8")
ui8 = FLAMEGPU.getVariableUInt8Array6("ui8")
c = FLAMEGPU.getVariableCharArray7("c")
uc = FLAMEGPU.getVariableUCharArray8("uc")
i16 = FLAMEGPU.getVariableInt16Array9("i16")
ui16 = FLAMEGPU.getVariableUInt16Array10("ui16")
i32 = FLAMEGPU.getVariableInt32Array11("i32")
ui32 = FLAMEGPU.getVariableUInt32Array12("ui32")
i64 = FLAMEGPU.getVariableInt64Array13("i64")
ui64 = FLAMEGPU.getVariableUInt64Array14("ui64")
"""

cpp_fgpu_array_types = """\
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
"""

py_fgpu_unknown_type = """\
i = FLAMEGPU.getVariableUnknownType("i")
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


py_fgpu_agent_func = """\
@flamegpu_agent_function
def func(message_in: MessageNone, message_out: MessageBruteForce):
    pass
"""
cpp_fgpu_agent_func = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageBruteForce){
    ;
}
"""

py_fgpu_agent_func_no_input_msg_type = """\
@flamegpu_agent_function
def func(message_in, message_out: MessageBruteForce):
    pass
"""

py_fgpu_agent_func_no_output_msg_type = """\
@flamegpu_agent_function
def func(message_in: MessageNone, message_out):
    pass
"""

py_fgpu_agent_func_extra_args = """\
@flamegpu_agent_function
def func(message_in: MessageBruteForce, message_out: MessageBruteForce, other):
    pass
"""

py_fgpu_agent_func_extra_args = """\
@flamegpu_agent_function
def func(message_in: MessageBruteForce, message_out: MessageBruteForce, other):
    pass
"""

py_fgpu_agent_func_return_type = """\
@flamegpu_agent_function
def func(message_in: MessageNone, message_out: MessageBruteForce) -> int :
    pass
"""
cpp_fgpu_agent_func_return_type = """\
FLAMEGPU_AGENT_FUNCTION(func, flamegpu::MessageNone, flamegpu::MessageBruteForce){
    ;
}
"""

py_fgpu_device_func_args = """\
@flamegpu_device_function
def func(x: int) -> int :
    pass
"""

cpp_fgpu_device_func_args = """\
FLAMEGPU_DEVICE_FUNCTION int func(int x){
    ;
}
"""

py_fgpu_device_func_no_type = """\
@flamegpu_device_function
def func(x) -> int :
    pass
"""

py_fgpu_device_func_no_return_type = """\
@flamegpu_device_function
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
@flamegpu_device_function
def func(x: int) :
    x = 10
"""
cpp_fgpu_device_func_arg_modified = """\
FLAMEGPU_DEVICE_FUNCTION void func(int x){
    x = 10;
}
"""

py_fgpu_device_local_args_stack = """\
@flamegpu_device_function
def funcA(x: int) :
    a = 1
    x = 10

@flamegpu_device_function
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
        self._checkException(py_for_unsupported, "Range based for loops only support")
       
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
        #self._checkExpected("-1e1000")
        #self._checkExpected("1e1000j")
        #self._checkExpected("-1e1000j")

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
        self._checkException("b'123'", "Byte strings not supported")

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


    def test_with(self):
        self._checkException(py_with_simple, "With not supported")
        self._checkException(py_with_as, "With not supported")
        self._checkException(py_async_with, "Async with not supported")
        self._checkException(py_async_with_as, "Async with not supported")

    def test_async_function_def(self):
        self._checkException(py_async_function_def, "Async functions not supported")

    def test_async_for(self):
        self._checkException(py_async_for, "Async for not supported")



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
        self._checkException("a: numpy.unsupported", "Not a supported numpy type")
        
    
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

    

    
    # message output
    
    def test_fgpu_msg_output(self):
        # Test message output 
        self._checkExpected(py_fgpu_msg_output, cpp_fgpu_msg_output)
        # Test message output with unknown function
        self._checkException("message_out.unsupported()", "Function 'unsupported' does not exist") 
        
    # random
    
    def test_fgpu_random(self):
        self._checkExpected("FLAMEGPU.random.uniformFloat()", "FLAMEGPU->random.uniform<float>();")
        self._checkExpected("FLAMEGPU.random.uniformDouble()", "FLAMEGPU->random.uniform<double>();")
        self._checkExpected("FLAMEGPU.random.uniformInt()", "FLAMEGPU->random.uniform<int>();")
        self._checkException("FLAMEGPU.random.uniformBadType()", "'BadType' is not a valid FLAME GPU type")
        self._checkException("FLAMEGPU.random.unsupported()", "Function 'unsupported' does not exist in FLAMEGPU.random")
    
    # environment    
    def test_fgpu_environment(self):
        self._checkExpected("FLAMEGPU.environment.getPropertyFloat('f')", 'FLAMEGPU->environment.getProperty<float>("f");')
        self._checkExpected("FLAMEGPU.environment.getPropertyDouble('d')", 'FLAMEGPU->environment.getProperty<double>("d");')
        self._checkExpected("FLAMEGPU.environment.getPropertyInt('i')", 'FLAMEGPU->environment.getProperty<int>("i");')
        self._checkExpected("FLAMEGPU.environment.containsProperty('p')", 'FLAMEGPU->environment.containsProperty("p");')
        self._checkException("FLAMEGPU.environment.getPropertyBadType()", "'BadType' is not a valid FLAME GPU type")
        self._checkException("FLAMEGPU.environment.unsupported()", "Function 'unsupported' does not exist in FLAMEGPU.environment")
    
    # agent functions, args renaming, arg types
    
    def test_fgpu_agent_func(self):
        self._checkExpected(py_fgpu_agent_func, cpp_fgpu_agent_func)
        
    def test_fgpu_agent_func_input_types(self):
        """ Try all the message input types by using a string replacement """
        # try all correct types
        for msg_type in pyflamegpu.codegen.CodeGenerator.fgpu_message_types:
            py_func = py_fgpu_agent_func.replace("MessageNone", msg_type)
            cpp_output = cpp_fgpu_agent_func.replace("MessageNone", msg_type)
            self._checkExpected(py_func, cpp_output)
        # try an incorrect type
        py_func = py_fgpu_agent_func.replace("MessageNone", "UnsupportedMessageType")
        self._checkException(py_func, "Message input type annotation not a supported message type")
        
    def test_fgpu_agent_func_output_types(self):
        """ Try all the message output types by using a string replacement """
        # try all correct types
        for msg_type in pyflamegpu.codegen.CodeGenerator.fgpu_message_types:
            py_func = py_fgpu_agent_func.replace("MessageBruteForce", msg_type)
            cpp_output = cpp_fgpu_agent_func.replace("MessageBruteForce", msg_type)
            self._checkExpected(py_func, cpp_output)
        # try an incorrect type
        py_func = py_fgpu_agent_func.replace("MessageBruteForce", "UnsupportedMessageType")
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
        
    # device functions, arg types and calling
    
    def test_fgpu_device_func(self):
        self._checkExpected(py_fgpu_device_func_args, cpp_fgpu_device_func_args)
        # function argument requires a type
        self._checkException(py_fgpu_device_func_no_type, "Device function argument requires type annotation")
        # no function return type should use void
        self._checkExpected(py_fgpu_device_func_no_return_type, cpp_fgpu_device_func_no_return_type)
        # function requires decorator
        self._checkException(py_fgpu_device_func_no_decorator, "Function definitions require a single FLAMEGPU decorator of")
        

    def test_alive_dead(self):
        self._checkExpected(str("FLAMEGPU.ALIVE"), "flamegpu::ALIVE;")
        self._checkExpected(str("FLAMEGPU.DEAD"), "flamegpu::DEAD;")
    