#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "flamegpu/flamegpu.h"

namespace flamegpu {

namespace test_runplan_io {
const char *XML_FILE_NAME = "test.xml";
const char *JSON_FILE_NAME = "test.json";
const char *BIN_FILE_NAME = "test.bin";
FLAMEGPU_EXIT_CONDITION(allow_0_steps) {
    return flamegpu::EXIT;
}
class MiniSim {
    std::string test_file;

 public:
    ~MiniSim() {
        // Cleanup
        if (!test_file.empty())
            ::remove(test_file.c_str());
    }
    void run(const std::string &test_file_name) {
        this->test_file = test_file_name;
        // Assertions for limits
        ASSERT_TRUE(std::numeric_limits<float>::has_quiet_NaN);
        ASSERT_TRUE(std::numeric_limits<float>::has_signaling_NaN);
        ASSERT_TRUE(std::numeric_limits<double>::has_quiet_NaN);
        ASSERT_TRUE(std::numeric_limits<double>::has_signaling_NaN);
        // Model description
        ModelDescription model("test_model");
        AgentDescription a = model.newAgent("a");
        model.addExitCondition(allow_0_steps);
        {
            EnvironmentDescription e = model.Environment();
            // Ensure default values differ!!!
            e.newProperty<float>("float", 112.0f);
            e.newProperty<double>("double", 113.0);
            e.newProperty<int64_t>("int64_t", 114);
            e.newProperty<uint64_t>("uint64_t", 115u);
            e.newProperty<int32_t>("int32_t", 116);
            e.newProperty<uint32_t>("uint32_t", 117u);
            e.newProperty<int16_t>("int16_t", 118);
            e.newProperty<uint16_t>("uint16_t", 119u);
            e.newProperty<int8_t>("int8_t", 120);
            e.newProperty<uint8_t>("uint8_t", 121u);
            e.newProperty<float, 3>("float_a", { 112.0f, 10.0f, 11.0f });
            e.newProperty<double, 3>("double_a", { 113.0, 10.0, 11.0 });
            e.newProperty<int64_t, 3>("int64_t_a", { 114, 10, 11 });
            e.newProperty<uint64_t, 3>("uint64_t_a", { 115u, 10u, 11u });
            e.newProperty<int32_t, 3>("int32_t_a", { 116, 10, 11 });
            e.newProperty<uint32_t, 3>("uint32_t_a", { 117u, 10u, 11u });
            e.newProperty<int16_t, 3>("int16_t_a", { 118, 10, 11});
            e.newProperty<uint16_t, 3>("uint16_t_a", { 119u, 10u, 11u });
            e.newProperty<int8_t, 3>("int8_t_a", { 120, 10, 11 });
            e.newProperty<uint8_t, 3>("uint8_t_a", {121u, 10u, 11u});
            // Limit values
            e.newProperty<float>("float_qnan", 0);
            e.newProperty<float>("float_snan", 0);
            e.newProperty<float>("float_inf", 0);
            e.newProperty<float>("float_inf_neg", 0);
            e.newProperty<double>("double_qnan", 0);
            e.newProperty<double>("double_snan", 0);
            e.newProperty<double>("double_inf", 0);
            e.newProperty<double>("double_inf_neg", 0);
        }
        {  // Run export
            RunPlanVector rpv(model, 3);
            {
                auto& rp = rpv[0];
                rp.setOutputSubdirectory("foo");
                rp.setRandomSimulationSeed(23);
                rp.setSteps(22);
                // Scalars
                rp.setProperty<float>("float", 12.0f);
                rp.setProperty<double>("double", 13.0);
                rp.setProperty<int64_t>("int64_t", 14);
                rp.setProperty<uint64_t>("uint64_t", 15u);
                rp.setProperty<int32_t>("int32_t", 16);
                rp.setProperty<uint32_t>("uint32_t", 17u);
                rp.setProperty<int16_t>("int16_t", 18);
                rp.setProperty<float, 3>("float_a", { 12.0f, 0.0f, 1.0f });  // Edge case
                rp.setProperty<uint16_t>("uint16_t", 19u);
                rp.setProperty<int8_t>("int8_t", 20);
                rp.setProperty<uint8_t>("uint8_t", 21u);
            }
            {
                auto& rp = rpv[1];
                rp.setOutputSubdirectory("BAR");
                rp.setRandomSimulationSeed(std::numeric_limits<uint64_t>::max());
                rp.setSteps(std::numeric_limits<uint32_t>::max());
                // Limit values
                rp.setProperty<float>("float_qnan", std::numeric_limits<float>::quiet_NaN());
                rp.setProperty<float>("float_snan", std::numeric_limits<float>::signaling_NaN());
                rp.setProperty<float>("float_inf", std::numeric_limits<float>::infinity());
                rp.setProperty<float>("float_inf_neg", -std::numeric_limits<float>::infinity());
                rp.setProperty<double>("double_qnan", std::numeric_limits<double>::quiet_NaN());
                rp.setProperty<double>("double_snan", std::numeric_limits<double>::quiet_NaN());
                rp.setProperty<double>("double_inf", std::numeric_limits<double>::infinity());
                rp.setProperty<double>("double_inf_neg", -std::numeric_limits<double>::infinity());
            }
            {
                auto& rp = rpv[2];
                rp.setOutputSubdirectory("FOObar");
                rp.setRandomSimulationSeed(0);
                rp.setSteps(0);
                // Arrays
                rp.setProperty<float, 3>("float_a", { 12.0f, 0.0f, 1.0f });
                rp.setProperty<double, 3>("double_a", { 13.0, 0.0, 1.0 });
                rp.setProperty<int64_t, 3>("int64_t_a", { 14, 0, 1 });
                rp.setProperty<uint64_t, 3>("uint64_t_a", { 15u, 0u, 1u });
                rp.setProperty<int32_t, 3>("int32_t_a", { 16, 0, 1 });
                rp.setProperty<float>("float", 12.0f);  // Edge case
                rp.setProperty<uint32_t, 3>("uint32_t_a", { 17u, 0u, 1u });
                rp.setProperty<int16_t, 3>("int16_t_a", { 18, 0, 1 });
                rp.setProperty<uint16_t, 3>("uint16_t_a", { 19u, 0u, 1u });
                rp.setProperty<int8_t, 3>("int8_t_a", { 20, 0, 1 });
                rp.setProperty<uint8_t, 3>("uint8_t_a", { 21u, 0u, 1u });
            }
            io::JSONRunPlanWriter::save(rpv, test_file_name);
        }
        {   // Run Import
            RunPlanVector rpv = io::JSONRunPlanReader::load(test_file_name, model);
            // Validate config matches
            {
                auto& rp = rpv[0];
                EXPECT_EQ(rp.getOutputSubdirectory(), "foo");
                EXPECT_EQ(rp.getRandomSimulationSeed(), 23u);
                EXPECT_EQ(rp.getSteps(), 22u);
                // Scalars
                EXPECT_EQ(rp.getProperty<float>("float"), 12.0f);
                EXPECT_EQ(rp.getProperty<double>("double"), 13.0);
                EXPECT_EQ(rp.getProperty<int64_t>("int64_t"), 14);
                EXPECT_EQ(rp.getProperty<uint64_t>("uint64_t"), 15u);
                EXPECT_EQ(rp.getProperty<int32_t>("int32_t"), 16);
                EXPECT_EQ(rp.getProperty<uint32_t>("uint32_t"), 17u);
                EXPECT_EQ(rp.getProperty<int16_t>("int16_t"), 18);
                EXPECT_EQ(rp.getProperty<uint16_t>("uint16_t"), 19u);
                EXPECT_EQ(rp.getProperty<int8_t>("int8_t"), 20);
                EXPECT_EQ(rp.getProperty<uint8_t>("uint8_t"), 21u);
                // Edge case
                const bool float_array_eq = rp.getProperty<float, 3>("float_a") == std::array<float, 3>{ 12.0f, 0.0f, 1.0f };
                EXPECT_TRUE(float_array_eq);
            }
            {
                auto& rp = rpv[1];
                EXPECT_EQ(rp.getOutputSubdirectory(), "BAR");
                EXPECT_EQ(rp.getRandomSimulationSeed(), std::numeric_limits<uint64_t>::max());
                EXPECT_EQ(rp.getSteps(), std::numeric_limits<uint32_t>::max());
                // Limits
                EXPECT_TRUE(std::isnan(rp.getProperty<float>("float_qnan")));
                EXPECT_TRUE(std::isnan(rp.getProperty<float>("float_snan")));
                EXPECT_EQ(rp.getProperty<float>("float_inf"), std::numeric_limits<float>::infinity());
                EXPECT_EQ(rp.getProperty<float>("float_inf_neg"), -std::numeric_limits<float>::infinity());
                EXPECT_TRUE(std::isnan(rp.getProperty<double>("double_qnan")));
                EXPECT_TRUE(std::isnan(rp.getProperty<double>("double_snan")));
                EXPECT_EQ(rp.getProperty<double>("double_inf"), std::numeric_limits<double>::infinity());
                EXPECT_EQ(rp.getProperty<double>("double_inf_neg"), -std::numeric_limits<double>::infinity());
            }
            {
                auto& rp = rpv[2];
                EXPECT_EQ(rp.getOutputSubdirectory(), "FOObar");
                EXPECT_EQ(rp.getRandomSimulationSeed(), 0u);
                EXPECT_EQ(rp.getSteps(), 0u);
                // Arrays
                const bool float_array_eq = rp.getProperty<float, 3>("float_a") == std::array<float, 3>{ 12.0f, 0.0f, 1.0f };
                EXPECT_TRUE(float_array_eq);
                const bool double_array_eq = rp.getProperty<double, 3>("double_a") == std::array<double, 3>{ 13.0, 0.0, 1.0 };
                EXPECT_TRUE(double_array_eq);
                const bool int64_t_array_eq = rp.getProperty<int64_t, 3>("int64_t_a") == std::array<int64_t, 3>{ 14, 0, 1 };
                EXPECT_TRUE(int64_t_array_eq);
                const bool uint64_t_array_eq = rp.getProperty<uint64_t, 3>("uint64_t_a") == std::array<uint64_t, 3>{ 15u, 0u, 1u };
                EXPECT_TRUE(uint64_t_array_eq);
                const bool int32_t_array_eq = rp.getProperty<int32_t, 3>("int32_t_a") == std::array<int32_t, 3>{ 16, 0, 1 };
                EXPECT_TRUE(int32_t_array_eq);
                const bool uint32_t_array_eq = rp.getProperty<uint32_t, 3>("uint32_t_a") == std::array<uint32_t, 3> { 17u, 0u, 1u };
                EXPECT_TRUE(uint32_t_array_eq);
                const bool int16_t_array_eq = rp.getProperty<int16_t, 3>("int16_t_a") == std::array<int16_t, 3>{ 18, 0, 1 };
                EXPECT_TRUE(int16_t_array_eq);
                const bool uint16_t_array_eq = rp.getProperty<uint16_t, 3>("uint16_t_a") == std::array<uint16_t, 3>{ 19u, 0u, 1u };
                EXPECT_TRUE(uint16_t_array_eq);
                const bool int8_t_array_eq = rp.getProperty<int8_t, 3>("int8_t_a") == std::array<int8_t, 3>{ 20, 0, 1 };
                EXPECT_TRUE(int8_t_array_eq);
                const bool uint8_t_array_eq = rp.getProperty<uint8_t, 3>("uint8_t_a") == std::array<uint8_t, 3>{ 21u, 0u, 1u };
                EXPECT_TRUE(uint8_t_array_eq);
                // Edge case
                EXPECT_EQ(rp.getProperty<float>("float"), 12.0f);
            }
            EXPECT_EQ(rpv.size(), 3u);
        }
        // Cleanup
        ASSERT_EQ(::remove(test_file_name.c_str()), 0);
    }
};

/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class TestRunPlanIO : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};
/* TEST IS CURRENTLY HARDCODED TO USE JSON IO
TEST_F(TestRunPlanIO, XML_WriteRead) {
    // Avoid fail if previous run didn't cleanup properly
    ::remove(XML_FILE_NAME);
    ms->run(XML_FILE_NAME);
}*/
TEST_F(TestRunPlanIO, JSON_WriteRead) {
    // Avoid fail if previous run didn't cleanup properly
    ::remove(JSON_FILE_NAME);
    ms->run(JSON_FILE_NAME);
}
}  // namespace test_runplan_io
}  // namespace flamegpu
