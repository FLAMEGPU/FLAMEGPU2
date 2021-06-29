import pytest
from unittest import TestCase
from pyflamegpu import *


ARRAY_TEST_LEN = 8

def AddGet_SetGet_test(type: str):
    """
    Function replicates the template version in C++ by getting type qualified add, get and set 
    functions from EnvironmentDescription
    """
    ed = pyflamegpu.EnvironmentDescription()
    add_func = getattr(ed, f"newProperty{type}")
    get_func = getattr(ed, f"getProperty{type}")
    set_func = getattr(ed, f"setProperty{type}")
    add_func("a", 1)
    assert get_func("a") == 1
    assert set_func("a", 2) == 1
    assert get_func("a") == 2



def AddGet_SetGet_array_test(type: str):
    """
    Function replicates the template version in C++ by getting type and array size qualified 
    add, get and set functions from EnvironmentDescription
    """
    ed = pyflamegpu.EnvironmentDescription()
    add_func = getattr(ed, f"newPropertyArray{type}")
    get_func = getattr(ed, f"getPropertyArray{type}")
    set_func = getattr(ed, f"setPropertyArray{type}")
    b = [0] * ARRAY_TEST_LEN
    c = [0] * ARRAY_TEST_LEN
    for i in range(ARRAY_TEST_LEN):
        b[i] = i
        c[i] = ARRAY_TEST_LEN-i
    add_func("a", ARRAY_TEST_LEN, b)
    a = get_func("a")
    for i in range(ARRAY_TEST_LEN):
        assert a[i] == b[i]
    set_func("a", c)
    for i in range(ARRAY_TEST_LEN):
        assert a[i] == b[i]
    a = get_func("a")
    for i in range(ARRAY_TEST_LEN):
        assert a[i] == c[i]


def AddGet_SetGet_array_element_test(type: str):
    """
    Function replicates the template version in C++ by getting type and array size qualified 
    add, get and set functions from EnvironmentDescription
    """
    ed = pyflamegpu.EnvironmentDescription()
    add_func = getattr(ed, f"newPropertyArray{type}")
    get_func = getattr(ed, f"getProperty{type}")
    set_func = getattr(ed, f"setProperty{type}")
    b = [0] * ARRAY_TEST_LEN
    c = [0] * ARRAY_TEST_LEN
    for i in range(ARRAY_TEST_LEN):
        b[i] = i
        c[i] = ARRAY_TEST_LEN-i
    add_func("a", ARRAY_TEST_LEN, b)
    for i in range(ARRAY_TEST_LEN):
        assert get_func("a", i) == b[i]
        assert set_func("a", i, c[i]) == b[i]
    for i in range(ARRAY_TEST_LEN):
        assert get_func("a", i) == c[i]


def ExceptionPropertyType_test(type1: str, type2: str):
    """
    Function replicates the template version in C++ by getting type and array size qualified 
    add, get and set functions from EnvironmentDescription
    """
    ed = pyflamegpu.EnvironmentDescription()
    add_func_t1 = getattr(ed, f"newProperty{type1}")
    add_func_array_t1 = getattr(ed, f"newPropertyArray{type1}")
    set_func_t1 = getattr(ed, f"setProperty{type1}")
    set_func_t2 = getattr(ed, f"setProperty{type2}")
    set_func_array_t2 = getattr(ed, f"setPropertyArray{type2}")
  
    a_t1 = 1
    a_t2 = 1
    b_t1 = [0] * ARRAY_TEST_LEN
    b_t2 = [0] * ARRAY_TEST_LEN
    for i in range(ARRAY_TEST_LEN):
        b_t1[i] = i
        b_t2[i] = i
    add_func_t1("a", a_t1, True)
    add_func_array_t1("b", ARRAY_TEST_LEN, b_t1, True)
    
    with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        set_func_t2("a", a_t2)
    assert e.value.type() == "InvalidEnvPropertyType"
    with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        set_func_array_t2("b", b_t2)
    assert e.value.type() == "InvalidEnvPropertyType"
    with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        set_func_t2("b", 0, a_t2)
    assert e.value.type() == "InvalidEnvPropertyType"
        

def ExceptionPropertyLength_test(type: str):
    """
    Function replicates the template version in C++ by getting type and array size qualified 
    add, get and set functions from EnvironmentDescription
    """
    ed = pyflamegpu.EnvironmentDescription()
    add_func = getattr(ed, f"newPropertyArray{type}")
    set_func = getattr(ed, f"setPropertyArray{type}")
    
    b = [0] * ARRAY_TEST_LEN
    _b1 = [0] * 1
    _b2 = [0] * (ARRAY_TEST_LEN + 1)
    _b3 = [0] * ARRAY_TEST_LEN * 2

    add_func("a", ARRAY_TEST_LEN, b)
    with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        set_func("a", _b1)
    with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        set_func("a", _b2)
    with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        set_func("a", _b3)
    # Added extra case to ensure that the above TypeErrors are not a result of the set_func not being found
    set_func("a", b)

def ExceptionPropertyRange_test(type:str):
    ed = pyflamegpu.EnvironmentDescription()
    add_func = getattr(ed, f"newPropertyArray{type}")
    set_func = getattr(ed, f"setProperty{type}")
    get_func = getattr(ed, f"getProperty{type}")
    b = [0] * ARRAY_TEST_LEN

    add_func("a", ARRAY_TEST_LEN, b)
    c = 12

    for i in range(ARRAY_TEST_LEN):
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException)  as e:
            set_func("a", ARRAY_TEST_LEN + i, c)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            get_func("a", ARRAY_TEST_LEN + i)
        assert e.value.type() == "OutOfBoundsException"


class EnvironmentDescriptionTest(TestCase):
 
    def test_addget_setget_float(self):
        AddGet_SetGet_test("Float")
        
    def test_addget_setget_double(self):
        AddGet_SetGet_test("Double")

    def test_addget_setget_int8(self):
        AddGet_SetGet_test("Int8")

    def test_addget_setget_int16(self):
        AddGet_SetGet_test("Int16") 
        
    def test_addget_setget_int32(self):
        AddGet_SetGet_test("Int32") 
        
    def test_addget_setget_int64(self):
        AddGet_SetGet_test("Int64") 
        
    def test_addget_setget_uint8(self):
        AddGet_SetGet_test("UInt8")

    def test_addget_setget_uint16(self):
        AddGet_SetGet_test("UInt16") 
        
    def test_addget_setget_uint32(self):
        AddGet_SetGet_test("UInt32") 
        
    def test_addget_setget_uint64(self):
        AddGet_SetGet_test("UInt64") 
        
       

       
    def test_addget_setgetarray_float(self):
        AddGet_SetGet_array_test("Float")
        
    def test_addget_setgetarray_double(self):
        AddGet_SetGet_array_test("Double")

    def test_addget_setgetarray_int8(self):
        AddGet_SetGet_array_test("Int8")

    def test_addget_setgetarray_int16(self):
        AddGet_SetGet_array_test("Int16") 
        
    def test_addget_setgetarray_int32(self):
        AddGet_SetGet_array_test("Int32") 
        
    def test_addget_setgetarray_int64(self):
        AddGet_SetGet_array_test("Int64") 
        
    def test_addget_setgetarray_uint8(self):
        AddGet_SetGet_array_test("UInt8")

    def test_addget_setgetarray_uint16(self):
        AddGet_SetGet_array_test("UInt16") 
        
    def test_addget_setgetarray_uint32(self):
        AddGet_SetGet_array_test("UInt32") 
        
    def test_addget_setgetarray_uint64(self):
        AddGet_SetGet_array_test("UInt64") 




    def test_addget_setgetarray_element_float(self):
        AddGet_SetGet_array_element_test("Float")
        
    def test_addget_setgetarray_element_double(self):
        AddGet_SetGet_array_element_test("Double")

    def test_addget_setgetarray_element_int8(self):
        AddGet_SetGet_array_element_test("Int8")

    def test_addget_setgetarray_element_int16(self):
        AddGet_SetGet_array_element_test("Int16") 
        
    def test_addget_setgetarray_element_int32(self):
        AddGet_SetGet_array_element_test("Int32") 
        
    def test_addget_setgetarray_element_int64(self):
        AddGet_SetGet_array_element_test("Int64") 
        
    def test_addget_setgetarray_element_uint8(self):
        AddGet_SetGet_array_element_test("UInt8")

    def test_addget_setgetarray_element_uint16(self):
        AddGet_SetGet_array_element_test("UInt16") 
        
    def test_addget_setgetarray_element_uint32(self):
        AddGet_SetGet_array_element_test("UInt32") 
        
    def test_addget_setgetarray_element_uint64(self):
        AddGet_SetGet_array_element_test("UInt64") 
    



    def test_exception_property_type_float(self):
        ExceptionPropertyType_test("Float", "UInt64")
        
    def test_exception_property_type_double(self):
        ExceptionPropertyType_test("Double", "UInt64")
        
    def test_exception_property_type_int8(self):
        ExceptionPropertyType_test("Int8", "UInt64")
        
    def test_exception_property_type_uint8(self):
        ExceptionPropertyType_test("UInt8", "UInt64")
        
    def test_exception_property_type_int16(self):
        ExceptionPropertyType_test("Int16", "UInt64")
        
    def test_exception_property_type_uint16(self):
        ExceptionPropertyType_test("UInt16", "UInt64")
        
    def test_exception_property_type_int32(self):
        ExceptionPropertyType_test("Int32", "UInt64")
        
    def test_exception_property_type_uint32(self):
        ExceptionPropertyType_test("UInt32", "UInt64")
        
    def test_exception_property_type_int64(self):
        ExceptionPropertyType_test("Int64", "Float")
        
    def test_exception_property_type_uint64(self):
        ExceptionPropertyType_test("UInt64", "Float")
        
        
        
    def test_exception_property_length_float(self):
        ExceptionPropertyLength_test("Float")
        
    def test_exception_property_length_double(self):
        ExceptionPropertyLength_test("Double")

    def test_exception_property_length_int8(self):
        ExceptionPropertyLength_test("Int8")

    def test_exception_property_length_int16(self):
        ExceptionPropertyLength_test("Int16") 
        
    def test_exception_property_length_int32(self):
        ExceptionPropertyLength_test("Int32") 
        
    def test_exception_property_length_int64(self):
        ExceptionPropertyLength_test("Int64") 
        
    def test_exception_property_length_uint8(self):
        ExceptionPropertyLength_test("UInt8")

    def test_exception_property_length_uint16(self):
        ExceptionPropertyLength_test("UInt16") 
        
    def test_exception_property_length_uint32(self):
        ExceptionPropertyLength_test("UInt32") 
        
    def test_exception_property_length_uint64(self):
        ExceptionPropertyLength_test("UInt64") 
    

 
    def test_exception_property_range_float(self):
        ExceptionPropertyRange_test("Float")
        
    def test_exception_property_range_double(self):
        ExceptionPropertyRange_test("Double")

    def test_exception_property_range_int8(self):
        ExceptionPropertyRange_test("Int8")

    def test_exception_property_range_int16(self):
        ExceptionPropertyRange_test("Int16") 
        
    def test_exception_property_range_int32(self):
        ExceptionPropertyRange_test("Int32") 
        
    def test_exception_property_range_int64(self):
        ExceptionPropertyRange_test("Int64") 
        
    def test_exception_property_range_uint8(self):
        ExceptionPropertyRange_test("UInt8")

    def test_exception_property_range_uint16(self):
        ExceptionPropertyRange_test("UInt16") 
        
    def test_exception_property_range_uint32(self):
        ExceptionPropertyRange_test("UInt32") 
        
    def test_exception_property_range_uint64(self):
        ExceptionPropertyRange_test("UInt64") 


    def test_exception_property_doesnt_exist(self):
        ed = pyflamegpu.EnvironmentDescription()
        a = 12.0
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ed.getPropertyFloat("a")
        assert e.value.type() == "InvalidEnvProperty"
        ed.newPropertyFloat("a", a)
        assert ed.getPropertyFloat("a") == a
        # array version (get array functions dynamically to avoid hard coded ARRAY_TEST_LEN)
        add_int_array_func = getattr(ed, f"newPropertyArrayInt")
        get_int_array_func = getattr(ed, f"getPropertyArrayInt")
        get_float_array_func = getattr(ed, f"getPropertyArrayFloat")
        b = [0] * ARRAY_TEST_LEN
        add_int_array_func("b", ARRAY_TEST_LEN, b, False)
        get_int_array_func("b")
        ed.getPropertyInt("b", 1)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            get_float_array_func("c")
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ed.getPropertyFloat("c", 1)
        assert e.value.type() == "InvalidEnvProperty"

    def test_reserved_name(self):
        ed = pyflamegpu.EnvironmentDescription()
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ed.newPropertyInt("_", 1)
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ed.setPropertyInt("_", 1)
        assert e.value.type() == "ReservedName"
        # Array version
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ed.newPropertyArrayInt("_", 2, [ 1, 2 ], False)
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ed.setPropertyArrayInt("_", [ 1, 2 ])
        assert e.value.type() == "ReservedName"
