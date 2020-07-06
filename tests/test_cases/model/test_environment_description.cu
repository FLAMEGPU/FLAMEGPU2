/**
 * Tests of class: EnvironmentDescription
 * 
 * Tests cover:
 * > get() [per supported type, individual/array/element]
 * > set() [per supported type, individual/array/element, isConst]
 * > remove() [does it work for isConst]
 * > exception throwing
 * Implied tests: (Covered as a result of other tests)
 * > add() [per supported type, individual/array]
 */

#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"

namespace {
const int ARRAY_TEST_LEN = 5;

/**
 * Tests void EnvDesc::add<T>(const std::string &, const T&) 
 * Tests T EnvDesc::get<T>(const std::string &)
 * Tests T EnvDesc::set<T>(const std::string &, const T&) 
 * Tests T EnvDesc::get<T>(const std::string &)
 */
template<typename T>
void AddGet_SetGet_test() {
    EnvironmentDescription ed;
    T b = static_cast<T>(1);
    T c = static_cast<T>(2);
    ed.add<T>("a", b);
    EXPECT_EQ(ed.get<T>("a"), b);
    EXPECT_EQ(ed.set<T>("a", c), b);
    EXPECT_EQ(ed.get<T>("a"), c);
}
/**
 * Tests void EnvDesc::add<T, N>(const std::string &, const std::array<T, N>&)
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
    ed.add<T, ARRAY_TEST_LEN>("a", b);
    std::array<T, ARRAY_TEST_LEN> a;
    a = ed.get<T, ARRAY_TEST_LEN>("a");
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(a[i], b[i]);
    }
    a = ed.set<T, ARRAY_TEST_LEN>("a", c);
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(a[i], b[i]);
    }
    a = ed.get<T, ARRAY_TEST_LEN>("a");
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(a[i], c[i]);
    }
}
/**
 * Tests void EnvDesc::add<T, N>(const std::string &, const std::array<T, N>&)
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
    ed.add<T, ARRAY_TEST_LEN>("a", b);
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(ed.get<T>("a", i), b[i]);
        EXPECT_EQ(ed.set<T>("a", i, c[i]), b[i]);
    }
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        EXPECT_EQ(ed.get<T>("a", i), c[i]);
    }
}

template<typename T, typename _T>
void ExceptionPropertyType_test() {
    EnvironmentDescription ed;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &EnvironmentDescription::set<_T, ARRAY_TEST_LEN>;

    T a = static_cast<T>(1);
    _T _a = static_cast<_T>(1);
    std::array<T, ARRAY_TEST_LEN> b;
    std::array<_T, ARRAY_TEST_LEN> _b;
    for (int i = 0; i < ARRAY_TEST_LEN; ++i) {
        b[i] = static_cast<T>(i);
        _b[i] = static_cast<_T>(i);
    }
    ed.add<T>("a", a, true);
    ed.add<T, ARRAY_TEST_LEN>("b", b, true);
    EXPECT_THROW(ed.set<_T>("a", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(ed.set<_T>("b", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((ed.*setArray)("b", _b), InvalidEnvPropertyType);
    EXPECT_THROW(ed.set<_T>("b", 0, _a), InvalidEnvPropertyType);
}

template<typename T>
void ExceptionPropertyLength_test() {
    EnvironmentDescription ed;
    std::array<T, ARRAY_TEST_LEN> b;
    std::array<T, 1> _b1;
    std::array<T, ARRAY_TEST_LEN + 1> _b2;
    std::array<T, ARRAY_TEST_LEN * 2> _b3;
    ed.add<T, ARRAY_TEST_LEN>("a", b);
    EXPECT_THROW(ed.set<T>("a", _b1), OutOfBoundsException);
    EXPECT_THROW(ed.set<T>("a", _b2), OutOfBoundsException);
    EXPECT_THROW(ed.set<T>("a", _b3), OutOfBoundsException);
}

template<typename T>
void ExceptionPropertyRange_test() {
    std::array<T, ARRAY_TEST_LEN> b;
    EnvironmentDescription ed;
    ed.add<T, ARRAY_TEST_LEN>("a", b);
    T c = static_cast<T>(12);

    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(ed.set<T>("a", ARRAY_TEST_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(ed.get<T>("a", ARRAY_TEST_LEN + i), OutOfBoundsException);
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
    EXPECT_THROW(ed.get<float>("a"), InvalidEnvProperty);
    ed.add<float>("a", a);
    EXPECT_EQ(ed.get<float>("a"), a);
    // array version
    auto f = &EnvironmentDescription::get<int, 2>;
    EXPECT_THROW((ed.*f)("b"), InvalidEnvProperty);
    auto addArray = &EnvironmentDescription::add<int, ARRAY_TEST_LEN>;
    std::array<int, ARRAY_TEST_LEN> b;
    // EXPECT_NO_THROW(ed.add<int>("b", b));  // Doesn't build on Travis
    EXPECT_NO_THROW((ed.*addArray)("b", b, false));
    EXPECT_NO_THROW(ed.get<int>("b"));
    EXPECT_NO_THROW(ed.get<int>("b", 1));
}

TEST(EnvironmentDescriptionTest, reserved_name) {
    EnvironmentDescription ed;
    EXPECT_THROW(ed.add<int>("_", 1), ReservedName);
    EXPECT_THROW(ed.set<int>("_", 1), ReservedName);
    auto add = &EnvironmentDescription::add<int, 2>;
    auto set = &EnvironmentDescription::set<int, 2>;
    EXPECT_THROW((ed.*add)("_", { 1, 2 }, false), ReservedName);
    EXPECT_THROW((ed.*set)("_", { 1, 2 }), ReservedName);
}
