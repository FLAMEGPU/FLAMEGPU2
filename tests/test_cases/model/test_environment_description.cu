/**
 * Tests of class: EnvironmentDescription
 * 
 * Tests cover:
 * > get() [per supported type, individual/array/element]
 * > set() [per supported type, individual/array/element, isConst]
 * > remove() [does it work for isConst]
 * > exception throwing
 * Implied tests: (Covered as a result of other tests)
 * > newProperty() [per supported type, individual/array]
 */

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"
namespace flamegpu {


namespace {
const int ARRAY_TEST_LEN = 5;

/**
 * Tests void EnvDesc::newProperty<T>(const std::string &, const T&) 
 * Tests T EnvDesc::get<T>(const std::string &)
 * Tests T EnvDesc::set<T>(const std::string &, const T&) 
 * Tests T EnvDesc::get<T>(const std::string &)
 */
template<typename T>
void AddGet_SetGet_test() {
    EnvironmentDescription ed;
    T b = static_cast<T>(1);
    T c = static_cast<T>(2);
    ed.newProperty<T>("a", b);
    EXPECT_EQ(ed.getProperty<T>("a"), b);
    EXPECT_EQ(ed.setProperty<T>("a", c), b);
    EXPECT_EQ(ed.getProperty<T>("a"), c);
}
/**
 * Tests void EnvDesc::newProperty<T, N>(const std::string &, const std::array<T, N>&)
 * Tests std::array<T, N> EnvDesc::get<T, N>(const std::string &)
 * Tests std::array<T, N> EnvDesc::set<T, N>(const std::string &, const std::array<T, N>&)
 * Tests std::array<T, N> EnvDesc::get<T, N>(const std::string &)
 */
template<typename T>
void AddGet_SetGet_array_test() {
    EnvironmentDescription ed;
    std::array<T, ARRAY_TEST_LEN> b;
    std::array<T, ARRAY_TEST_LEN> c;
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        b[i] = static_cast<T>(i);
        c[i] = static_cast<T>(ARRAY_TEST_LEN-i);
    }
    ed.newProperty<T, ARRAY_TEST_LEN>("a", b);
    std::array<T, ARRAY_TEST_LEN> a;
    a = ed.getProperty<T, ARRAY_TEST_LEN>("a");
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(a[i], b[i]);
    }
    a = ed.setProperty<T, ARRAY_TEST_LEN>("a", c);
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(a[i], b[i]);
    }
    a = ed.getProperty<T, ARRAY_TEST_LEN>("a");
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(a[i], c[i]);
    }
}
/**
 * Tests void EnvDesc::newProperty<T, N>(const std::string &, const std::array<T, N>&)
 * Tests T EnvDesc::get<T, N>(const std::string &, const size_type &)
 * Tests T EnvDesc::set<T, N>(const std::string &, const size_type &, const T &)
 * Tests T EnvDesc::get<T, N>(const std::string &, const size_type &)
 */
template<typename T>
void AddGet_SetGet_array_element_test() {
    EnvironmentDescription ed;
    std::array<T, ARRAY_TEST_LEN> b;
    std::array<T, ARRAY_TEST_LEN> c;
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        b[i] = static_cast<T>(i);
        c[i] = static_cast<T>(ARRAY_TEST_LEN - i);
    }
    ed.newProperty<T, ARRAY_TEST_LEN>("a", b);
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(ed.getProperty<T>("a", i), b[i]);
        EXPECT_EQ(ed.setProperty<T>("a", i, c[i]), b[i]);
    }
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(ed.getProperty<T>("a", i), c[i]);
    }
}

template<typename T, typename _T>
void ExceptionPropertyType_test() {
    EnvironmentDescription ed;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &EnvironmentDescription::setProperty<_T, ARRAY_TEST_LEN>;

    T a = static_cast<T>(1);
    _T _a = static_cast<_T>(1);
    std::array<T, ARRAY_TEST_LEN> b;
    std::array<_T, ARRAY_TEST_LEN> _b;
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        b[i] = static_cast<T>(i);
        _b[i] = static_cast<_T>(i);
    }
    ed.newProperty<T>("a", a, true);
    ed.newProperty<T, ARRAY_TEST_LEN>("b", b, true);
    EXPECT_THROW(ed.setProperty<_T>("a", _a), exception::InvalidEnvPropertyType);
    // EXPECT_THROW(ed.set<_T>("b", _b), exception::InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((ed.*setArray)("b", _b), exception::InvalidEnvPropertyType);
    EXPECT_THROW(ed.setProperty<_T>("b", 0, _a), exception::InvalidEnvPropertyType);
}

template<typename T>
void ExceptionPropertyLength_test() {
    EnvironmentDescription ed;
    std::array<T, ARRAY_TEST_LEN> b;
    std::array<T, 1> _b1;
    std::array<T, ARRAY_TEST_LEN + 1> _b2;
    std::array<T, ARRAY_TEST_LEN * 2> _b3;
    ed.newProperty<T, ARRAY_TEST_LEN>("a", b);
    auto fn1 = &EnvironmentDescription::setProperty<T, 1>;
    auto fn2 = &EnvironmentDescription::setProperty<T, ARRAY_TEST_LEN + 1>;
    auto fn3 = &EnvironmentDescription::setProperty<T, ARRAY_TEST_LEN * 2>;
    EXPECT_THROW((ed.*fn1)("a", _b1), exception::OutOfBoundsException);
    EXPECT_THROW((ed.*fn2)("a", _b2), exception::OutOfBoundsException);
    EXPECT_THROW((ed.*fn3)("a", _b3), exception::OutOfBoundsException);
}

template<typename T>
void ExceptionPropertyRange_test() {
    std::array<T, ARRAY_TEST_LEN> b;
    EnvironmentDescription ed;
    ed.newProperty<T, ARRAY_TEST_LEN>("a", b);
    T c = static_cast<T>(12);

    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(ed.setProperty<T>("a", ARRAY_TEST_LEN + i, c), exception::OutOfBoundsException);
        EXPECT_THROW(ed.getProperty<T>("a", ARRAY_TEST_LEN + i), exception::OutOfBoundsException);
    }
}

}  // namespace


TEST(EnvironmentDescriptionTest, AddGet_SetGetfloat) {
    AddGet_SetGet_test<float>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetdouble) {
    AddGet_SetGet_test<double>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetint8_t) {
    AddGet_SetGet_test<int8_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetuint8_t) {
    AddGet_SetGet_test<uint8_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetint16_t) {
    AddGet_SetGet_test<int16_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetuint16_t) {
    AddGet_SetGet_test<uint16_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetint32_t) {
    AddGet_SetGet_test<int32_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetuint32_t) {
    AddGet_SetGet_test<uint32_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetint64_t) {
    AddGet_SetGet_test<int64_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetuint64_t) {
    AddGet_SetGet_test<uint64_t>();
}

TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_float) {
    AddGet_SetGet_array_test<float>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_double) {
    AddGet_SetGet_array_test<double>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_int8_t) {
    AddGet_SetGet_array_test<int8_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_uint8_t) {
    AddGet_SetGet_array_test<uint8_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_int16_t) {
    AddGet_SetGet_array_test<int16_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_uint16_t) {
    AddGet_SetGet_array_test<uint16_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_int32_t) {
    AddGet_SetGet_array_test<int32_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_uint32_t) {
    AddGet_SetGet_array_test<uint32_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_int64_t) {
    AddGet_SetGet_array_test<int64_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_uint64_t) {
    AddGet_SetGet_array_test<uint64_t>();
}

TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_float) {
    AddGet_SetGet_array_element_test<float>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_double) {
    AddGet_SetGet_array_element_test<double>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_int8_t) {
    AddGet_SetGet_array_element_test<int8_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_uint8_t) {
    AddGet_SetGet_array_element_test<uint8_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_int16_t) {
    AddGet_SetGet_array_element_test<int16_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_uint16_t) {
    AddGet_SetGet_array_element_test<uint16_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_int32_t) {
    AddGet_SetGet_array_element_test<int32_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_uint32_t) {
    AddGet_SetGet_array_element_test<uint32_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_int64_t) {
    AddGet_SetGet_array_element_test<int64_t>();
}
TEST(EnvironmentDescriptionTest, AddGet_SetGetarray_element_uint64_t) {
    AddGet_SetGet_array_element_test<uint64_t>();
}

TEST(EnvironmentDescriptionTest, ExceptionPropertyType_float) {
    ExceptionPropertyType_test<float, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_double) {
    ExceptionPropertyType_test<double, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_int8_t) {
    ExceptionPropertyType_test<int8_t, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_uint8_t) {
    ExceptionPropertyType_test<uint8_t, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_int16_t) {
    ExceptionPropertyType_test<int16_t, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_uint16_t) {
    ExceptionPropertyType_test<uint16_t, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_int32_t) {
    ExceptionPropertyType_test<int32_t, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_uint32_t) {
    ExceptionPropertyType_test<uint32_t, uint64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_int64_t) {
    ExceptionPropertyType_test<int64_t, float>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyType_uint64_t) {
    ExceptionPropertyType_test<uint64_t, float>();
}

TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_float) {
    ExceptionPropertyLength_test<float>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_double) {
    ExceptionPropertyLength_test<double>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_int8_t) {
    ExceptionPropertyLength_test<int8_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_uint8_t) {
    ExceptionPropertyLength_test<uint8_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_int16_t) {
    ExceptionPropertyLength_test<int16_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_uint16_t) {
    ExceptionPropertyLength_test<uint16_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_int32_t) {
    ExceptionPropertyLength_test<int32_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_uint32_t) {
    ExceptionPropertyLength_test<uint32_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_int64_t) {
    ExceptionPropertyLength_test<int64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyLength_uint64_t) {
    ExceptionPropertyLength_test<uint64_t>();
}

TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_float) {
    ExceptionPropertyRange_test<float>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_double) {
    ExceptionPropertyRange_test<double>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_int8_t) {
    ExceptionPropertyRange_test<int8_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_uint8_t) {
    ExceptionPropertyRange_test<uint8_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_int16_t) {
    ExceptionPropertyRange_test<int16_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_uint16_t) {
    ExceptionPropertyRange_test<uint16_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_int32_t) {
    ExceptionPropertyRange_test<int32_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_uint32_t) {
    ExceptionPropertyRange_test<uint32_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_int64_t) {
    ExceptionPropertyRange_test<int64_t>();
}
TEST(EnvironmentDescriptionTest, ExceptionPropertyRange_uint64_t) {
    ExceptionPropertyRange_test<uint64_t>();
}

TEST(EnvironmentDescriptionTest, ExceptionPropertyDoesntExist) {
    EnvironmentDescription ed;
    float a = static_cast<float>(12);
    EXPECT_THROW(ed.getProperty<float>("a"), exception::InvalidEnvProperty);
    ed.newProperty<float>("a", a);
    EXPECT_EQ(ed.getProperty<float>("a"), a);
    // array version
    auto f = &EnvironmentDescription::getProperty<int, 2>;
    EXPECT_THROW((ed.*f)("b"), exception::InvalidEnvProperty);
    auto addArray = &EnvironmentDescription::newProperty<int, ARRAY_TEST_LEN>;
    std::array<int, ARRAY_TEST_LEN> b;
    // EXPECT_NO_THROW(ed.newProperty<int>("b", b));  // Doesn't build on Travis
    EXPECT_NO_THROW((ed.*addArray)("b", b, false));
    EXPECT_NO_THROW(ed.getProperty<int>("b"));
    EXPECT_NO_THROW(ed.getProperty<int>("b", 1));
}

TEST(EnvironmentDescriptionTest, reserved_name) {
    EnvironmentDescription ed;
    EXPECT_THROW(ed.newProperty<int>("_", 1), exception::ReservedName);
    EXPECT_THROW(ed.setProperty<int>("_", 1), exception::ReservedName);
    auto add = &EnvironmentDescription::newProperty<int, 2>;
    auto set = &EnvironmentDescription::setProperty<int, 2>;
    EXPECT_THROW((ed.*add)("_", { 1, 2 }, false), exception::ReservedName);
    EXPECT_THROW((ed.*set)("_", { 1, 2 }), exception::ReservedName);
}
}  // namespace flamegpu
