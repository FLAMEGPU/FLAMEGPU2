/**
 * Tests of class: HostEnvironment
 * 
 * Tests cover:
 * > get() [per supported type, individual/array/element]
 * > set() [per supported type, individual/array/element]
 * > add() [per supported type, individual/array]
 * > remove() (implied by exception tests)
 * exceptions
 */

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace {
const unsigned int TEST_LEN = 256;
const int TEST_ARRAY_LEN = 5;
const uint8_t TEST_VALUE = 12;
const uint8_t TEST_ARRAY_OFFSET = 2;

class MiniSim {
 public:
    MiniSim() :
        model("model"),
        agent(model.newAgent("agent")),
        population(nullptr),
        ed(model.Environment()) {
        population = new AgentPopulation(agent, TEST_LEN);
        ed.add<float>("float_", static_cast<float>(TEST_VALUE));
        ed.add<double>("double_", static_cast<double>(TEST_VALUE));
        ed.add<int8_t>("int8_t_", static_cast<int8_t>(TEST_VALUE));
        ed.add<uint8_t>("uint8_t_", static_cast<uint8_t>(TEST_VALUE));
        ed.add<int16_t>("int16_t_", static_cast<int16_t>(TEST_VALUE));
        ed.add<uint16_t>("uint16_t_", static_cast<uint16_t>(TEST_VALUE));
        ed.add<int32_t>("int32_t_", static_cast<int32_t>(TEST_VALUE));
        ed.add<uint32_t>("uint32_t_", static_cast<uint32_t>(TEST_VALUE));
        ed.add<int64_t>("int64_t_", static_cast<int64_t>(TEST_VALUE));
        ed.add<uint64_t>("uint64_t_", static_cast<uint64_t>(TEST_VALUE));

        ed.add<float, TEST_ARRAY_LEN>("float_a_", makeInit<float>());
        ed.add<double, TEST_ARRAY_LEN>("double_a_", makeInit<double>());
        ed.add<int8_t, TEST_ARRAY_LEN>("int8_t_a_", makeInit<int8_t>());
        ed.add<uint8_t, TEST_ARRAY_LEN>("uint8_t_a_", makeInit<uint8_t>());
        ed.add<int16_t, TEST_ARRAY_LEN>("int16_t_a_", makeInit<int16_t>());
        ed.add<uint16_t, TEST_ARRAY_LEN>("uint16_t_a_", makeInit<uint16_t>());
        ed.add<int32_t, TEST_ARRAY_LEN>("int32_t_a_", makeInit<int32_t>());
        ed.add<uint32_t, TEST_ARRAY_LEN>("uint32_t_a_", makeInit<uint32_t>());
        ed.add<int64_t, TEST_ARRAY_LEN>("int64_t_a_", makeInit<int64_t>());
        ed.add<uint64_t, TEST_ARRAY_LEN>("uint64_t_a_", makeInit<uint64_t>());
    }
    ~MiniSim() { delete population;  }
    template <typename T>
    static std::array<T, TEST_ARRAY_LEN> makeInit(int offset = 0) {
        std::array<T, TEST_ARRAY_LEN> init;
        for (int i = 0; i < TEST_ARRAY_LEN; ++i)
            init[i] = static_cast<T>(i + 1 + offset);
        return init;
    }
    void run(int steps = 2) {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDAAgentModel cuda_model(model);
        cuda_model.setSimulationSteps(steps);
        // This fails as agentMap is empty
        cuda_model.setPopulationData(*population);
        ASSERT_NO_THROW(cuda_model.simulate());
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cuda_model.getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription &agent;
    AgentPopulation *population;
    EnvironmentDescription &ed;
};

FLAMEGPU_STEP_FUNCTION(add_get_set_float) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<float>("float_", static_cast<float>(TEST_VALUE) * 2), static_cast<float>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<float>("float_"), static_cast<float>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<float>("float", static_cast<float>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<float>("float", static_cast<float>(TEST_VALUE)), static_cast<float>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<float>("float"), static_cast<float>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<float>("float_", static_cast<float>(TEST_VALUE));
    FLAMEGPU->environment.remove<float>("float");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_double) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<double>("double_", static_cast<double>(TEST_VALUE) * 2), static_cast<double>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<double>("double_"), static_cast<double>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<double>("double", static_cast<double>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<double>("double", static_cast<double>(TEST_VALUE)), static_cast<double>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<double>("double"), static_cast<double>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<double>("double_", static_cast<double>(TEST_VALUE));
    FLAMEGPU->environment.remove<double>("double");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_int8_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int8_t>("int8_t_", static_cast<int8_t>(TEST_VALUE) * 2), static_cast<int8_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int8_t>("int8_t_"), static_cast<int8_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int8_t>("int8_t", static_cast<int8_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int8_t>("int8_t", static_cast<int8_t>(TEST_VALUE)), static_cast<int8_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int8_t>("int8_t"), static_cast<int8_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int8_t>("int8_t_", static_cast<int8_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<int8_t>("int8_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_uint8_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint8_t>("uint8_t_", static_cast<uint8_t>(TEST_VALUE) * 2), static_cast<uint8_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint8_t>("uint8_t_"), static_cast<uint8_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint8_t>("uint8_t", static_cast<uint8_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint8_t>("uint8_t", static_cast<uint8_t>(TEST_VALUE)), static_cast<uint8_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint8_t>("uint8_t"), static_cast<uint8_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint8_t>("uint8_t_", static_cast<uint8_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<uint8_t>("uint8_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_int16_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int16_t>("int16_t_", static_cast<int16_t>(TEST_VALUE) * 2), static_cast<int16_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int16_t>("int16_t_"), static_cast<int16_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int16_t>("int16_t", static_cast<int16_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int16_t>("int16_t", static_cast<int16_t>(TEST_VALUE)), static_cast<int16_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int16_t>("int16_t"), static_cast<int16_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int16_t>("int16_t_", static_cast<int16_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<int16_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_uint16_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint16_t>("uint16_t_", static_cast<uint16_t>(TEST_VALUE) * 2), static_cast<uint16_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint16_t>("uint16_t_"), static_cast<uint16_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint16_t>("uint16_t", static_cast<uint16_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint16_t>("uint16_t", static_cast<uint16_t>(TEST_VALUE)), static_cast<uint16_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint16_t>("uint16_t"), static_cast<uint16_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint16_t>("uint16_t_", static_cast<uint16_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<uint16_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_int32_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int32_t>("int32_t_", static_cast<int32_t>(TEST_VALUE) * 2), static_cast<int32_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int32_t>("int32_t_"), static_cast<int32_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int32_t>("int32_t", static_cast<int32_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int32_t>("int32_t", static_cast<int32_t>(TEST_VALUE)), static_cast<int32_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int32_t>("int32_t"), static_cast<int32_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int32_t>("int32_t_", static_cast<int32_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_uint32_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint32_t>("uint32_t_", static_cast<uint32_t>(TEST_VALUE) * 2), static_cast<uint32_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint32_t>("uint32_t_"), static_cast<uint32_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint32_t>("uint32_t", static_cast<uint32_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint32_t>("uint32_t", static_cast<uint32_t>(TEST_VALUE)), static_cast<uint32_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint32_t>("uint32_t"), static_cast<uint32_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint32_t>("uint32_t_", static_cast<uint32_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<uint32_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_int64_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int64_t>("int64_t_", static_cast<int64_t>(TEST_VALUE) * 2), static_cast<int64_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int64_t>("int64_t_"), static_cast<int64_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int64_t>("int64_t", static_cast<int64_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int64_t>("int64_t", static_cast<int64_t>(TEST_VALUE)), static_cast<int64_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int64_t>("int64_t"), static_cast<int64_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int64_t>("int64_t_", static_cast<int64_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<int64_t>("int64_t");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_uint64_t) {
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint64_t>("uint64_t_", static_cast<uint64_t>(TEST_VALUE) * 2), static_cast<uint64_t>(TEST_VALUE));
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint64_t>("uint64_t_"), static_cast<uint64_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint64_t>("uint64_t", static_cast<uint64_t>(TEST_VALUE) * 2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint64_t>("uint64_t", static_cast<uint64_t>(TEST_VALUE)), static_cast<uint64_t>(TEST_VALUE) * 2);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint64_t>("uint64_t"), static_cast<uint64_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint64_t>("uint64_t_", static_cast<uint64_t>(TEST_VALUE));
    FLAMEGPU->environment.remove<uint64_t>("uint64_t");
}

FLAMEGPU_STEP_FUNCTION(add_get_set_array_float) {
    std::array<float, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<float>();
    std::array<float, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<float>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<float, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<float, TEST_ARRAY_LEN>("float_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<float, TEST_ARRAY_LEN>("float_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<float, TEST_ARRAY_LEN>("float_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<float, TEST_ARRAY_LEN>("float_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<float, TEST_ARRAY_LEN>("float_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<float, TEST_ARRAY_LEN>("float_a_", init1);
    FLAMEGPU->environment.remove<float>("float_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_double) {
    std::array<double, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<double>();
    std::array<double, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<double>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<double, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<double, TEST_ARRAY_LEN>("double_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<double, TEST_ARRAY_LEN>("double_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<double, TEST_ARRAY_LEN>("double_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<double, TEST_ARRAY_LEN>("double_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<double, TEST_ARRAY_LEN>("double_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<double, TEST_ARRAY_LEN>("double_a_", init1);
    FLAMEGPU->environment.remove<double>("double_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_int8_t) {
    std::array<int8_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int8_t>();
    std::array<int8_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int8_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<int8_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<int8_t, TEST_ARRAY_LEN>("int8_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<int8_t, TEST_ARRAY_LEN>("int8_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<int8_t, TEST_ARRAY_LEN>("int8_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<int8_t, TEST_ARRAY_LEN>("int8_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<int8_t, TEST_ARRAY_LEN>("int8_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<int8_t, TEST_ARRAY_LEN>("int8_t_a_", init1);
    FLAMEGPU->environment.remove<int8_t>("int8_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_uint8_t) {
    std::array<uint8_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint8_t>();
    std::array<uint8_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint8_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<uint8_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<uint8_t, TEST_ARRAY_LEN>("uint8_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<uint8_t, TEST_ARRAY_LEN>("uint8_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<uint8_t, TEST_ARRAY_LEN>("uint8_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<uint8_t, TEST_ARRAY_LEN>("uint8_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<uint8_t, TEST_ARRAY_LEN>("uint8_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<uint8_t, TEST_ARRAY_LEN>("uint8_t_a_", init1);
    FLAMEGPU->environment.remove<uint8_t>("uint8_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_int16_t) {
    std::array<int16_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int16_t>();
    std::array<int16_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int16_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<int16_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<int16_t, TEST_ARRAY_LEN>("int16_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<int16_t, TEST_ARRAY_LEN>("int16_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<int16_t, TEST_ARRAY_LEN>("int16_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<int16_t, TEST_ARRAY_LEN>("int16_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<int16_t, TEST_ARRAY_LEN>("int16_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<int16_t, TEST_ARRAY_LEN>("int16_t_a_", init1);
    FLAMEGPU->environment.remove<int16_t>("int16_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_uint16_t) {
    std::array<uint16_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint16_t>();
    std::array<uint16_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint16_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<uint16_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<uint16_t, TEST_ARRAY_LEN>("uint16_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<uint16_t, TEST_ARRAY_LEN>("uint16_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<uint16_t, TEST_ARRAY_LEN>("uint16_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<uint16_t, TEST_ARRAY_LEN>("uint16_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<uint16_t, TEST_ARRAY_LEN>("uint16_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<uint16_t, TEST_ARRAY_LEN>("uint16_t_a_", init1);
    FLAMEGPU->environment.remove<uint16_t>("uint16_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_int32_t) {
    std::array<int32_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int32_t>();
    std::array<int32_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int32_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<int32_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<int32_t, TEST_ARRAY_LEN>("int32_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<int32_t, TEST_ARRAY_LEN>("int32_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<int32_t, TEST_ARRAY_LEN>("int32_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<int32_t, TEST_ARRAY_LEN>("int32_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<int32_t, TEST_ARRAY_LEN>("int32_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<int32_t, TEST_ARRAY_LEN>("int32_t_a_", init1);
    FLAMEGPU->environment.remove<int32_t>("int32_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_uint32_t) {
    std::array<uint32_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint32_t>();
    std::array<uint32_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint32_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<uint32_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<uint32_t, TEST_ARRAY_LEN>("uint32_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<uint32_t, TEST_ARRAY_LEN>("uint32_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<uint32_t, TEST_ARRAY_LEN>("uint32_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<uint32_t, TEST_ARRAY_LEN>("uint32_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<uint32_t, TEST_ARRAY_LEN>("uint32_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<uint32_t, TEST_ARRAY_LEN>("uint32_t_a_", init1);
    FLAMEGPU->environment.remove<uint32_t>("uint32_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_int64_t) {
    std::array<int64_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int64_t>();
    std::array<int64_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int64_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<int64_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<int64_t, TEST_ARRAY_LEN>("int64_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<int64_t, TEST_ARRAY_LEN>("int64_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<int64_t, TEST_ARRAY_LEN>("int64_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<int64_t, TEST_ARRAY_LEN>("int64_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<int64_t, TEST_ARRAY_LEN>("int64_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<int64_t, TEST_ARRAY_LEN>("int64_t_a_", init1);
    FLAMEGPU->environment.remove<int64_t>("int64_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_uint64_t) {
    std::array<uint64_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint64_t>();
    std::array<uint64_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint64_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    std::array<uint64_t, TEST_ARRAY_LEN> t = FLAMEGPU->environment.set<uint64_t, TEST_ARRAY_LEN>("uint64_t_a_", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get (Host func set value)
    t = FLAMEGPU->environment.get<uint64_t, TEST_ARRAY_LEN>("uint64_t_a_");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Add
    FLAMEGPU->environment.add<uint64_t, TEST_ARRAY_LEN>("uint64_t_a", init1);
    // Test Set (Host func added value)
    t = FLAMEGPU->environment.set<uint64_t, TEST_ARRAY_LEN>("uint64_t_a", init2);
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init1[i]);
    }
    // Test Get
    t = FLAMEGPU->environment.get<uint64_t, TEST_ARRAY_LEN>("uint64_t_a");
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        EXPECT_EQ(t[i], init2[i]);
    }
    // Reset for next iteration
    FLAMEGPU->environment.set<uint64_t, TEST_ARRAY_LEN>("uint64_t_a_", init1);
    FLAMEGPU->environment.remove<uint64_t>("uint64_t_a");
}

FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_float) {
    std::array<float, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<float>();
    std::array<float, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<float>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<float>("float_a_", TEST_ARRAY_LEN - 1, static_cast<float>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<float>("float_a_", TEST_ARRAY_LEN - 1), static_cast<float>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<float, TEST_ARRAY_LEN>("float_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<float>("float_a", TEST_ARRAY_LEN - 1, static_cast<float>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<float>("float_a", TEST_ARRAY_LEN - 1), static_cast<float>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<float>("float_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<float>("float_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_double) {
    std::array<double, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<double>();
    std::array<double, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<double>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<double>("double_a_", TEST_ARRAY_LEN - 1, static_cast<double>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<double>("double_a_", TEST_ARRAY_LEN - 1), static_cast<double>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<double, TEST_ARRAY_LEN>("double_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<double>("double_a", TEST_ARRAY_LEN - 1, static_cast<double>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<double>("double_a", TEST_ARRAY_LEN - 1), static_cast<double>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<double>("double_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<double>("double_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_int8_t) {
    std::array<int8_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int8_t>();
    std::array<int8_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int8_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int8_t>("int8_t_a_", TEST_ARRAY_LEN - 1, static_cast<int8_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int8_t>("int8_t_a_", TEST_ARRAY_LEN - 1), static_cast<int8_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int8_t, TEST_ARRAY_LEN>("int8_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int8_t>("int8_t_a", TEST_ARRAY_LEN - 1, static_cast<int8_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int8_t>("int8_t_a", TEST_ARRAY_LEN - 1), static_cast<int8_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int8_t>("int8_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<int8_t>("int8_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_uint8_t) {
    std::array<uint8_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint8_t>();
    std::array<uint8_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint8_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint8_t>("uint8_t_a_", TEST_ARRAY_LEN - 1, static_cast<uint8_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint8_t>("uint8_t_a_", TEST_ARRAY_LEN - 1), static_cast<uint8_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint8_t, TEST_ARRAY_LEN>("uint8_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint8_t>("uint8_t_a", TEST_ARRAY_LEN - 1, static_cast<uint8_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint8_t>("uint8_t_a", TEST_ARRAY_LEN - 1), static_cast<uint8_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint8_t>("uint8_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<uint8_t>("uint8_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_int16_t) {
    std::array<int16_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int16_t>();
    std::array<int16_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int16_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int16_t>("int16_t_a_", TEST_ARRAY_LEN - 1, static_cast<int16_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int16_t>("int16_t_a_", TEST_ARRAY_LEN - 1), static_cast<int16_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int16_t, TEST_ARRAY_LEN>("int16_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int16_t>("int16_t_a", TEST_ARRAY_LEN - 1, static_cast<int16_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int16_t>("int16_t_a", TEST_ARRAY_LEN - 1), static_cast<int16_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int16_t>("int16_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<int16_t>("int16_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_uint16_t) {
    std::array<uint16_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint16_t>();
    std::array<uint16_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint16_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint16_t>("uint16_t_a_", TEST_ARRAY_LEN - 1, static_cast<uint16_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint16_t>("uint16_t_a_", TEST_ARRAY_LEN - 1), static_cast<uint16_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint16_t, TEST_ARRAY_LEN>("uint16_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint16_t>("uint16_t_a", TEST_ARRAY_LEN - 1, static_cast<uint16_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint16_t>("uint16_t_a", TEST_ARRAY_LEN - 1), static_cast<uint16_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint16_t>("uint16_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<uint16_t>("uint16_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_int32_t) {
    std::array<int32_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int32_t>();
    std::array<int32_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int32_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int32_t>("int32_t_a_", TEST_ARRAY_LEN - 1, static_cast<int32_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int32_t>("int32_t_a_", TEST_ARRAY_LEN - 1), static_cast<int32_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int32_t, TEST_ARRAY_LEN>("int32_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int32_t>("int32_t_a", TEST_ARRAY_LEN - 1, static_cast<int32_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int32_t>("int32_t_a", TEST_ARRAY_LEN - 1), static_cast<int32_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int32_t>("int32_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<int32_t>("int32_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_uint32_t) {
    std::array<uint32_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint32_t>();
    std::array<uint32_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint32_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint32_t>("uint32_t_a_", TEST_ARRAY_LEN - 1, static_cast<uint32_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint32_t>("uint32_t_a_", TEST_ARRAY_LEN - 1), static_cast<uint32_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint32_t, TEST_ARRAY_LEN>("uint32_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint32_t>("uint32_t_a", TEST_ARRAY_LEN - 1, static_cast<uint32_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint32_t>("uint32_t_a", TEST_ARRAY_LEN - 1), static_cast<uint32_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint32_t>("uint32_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<uint32_t>("uint32_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_int64_t) {
    std::array<int64_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<int64_t>();
    std::array<int64_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<int64_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<int64_t>("int64_t_a_", TEST_ARRAY_LEN - 1, static_cast<int64_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<int64_t>("int64_t_a_", TEST_ARRAY_LEN - 1), static_cast<int64_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<int64_t, TEST_ARRAY_LEN>("int64_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<int64_t>("int64_t_a", TEST_ARRAY_LEN - 1, static_cast<int64_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<int64_t>("int64_t_a", TEST_ARRAY_LEN - 1), static_cast<int64_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<int64_t>("int64_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<int64_t>("int64_t_a");
}
FLAMEGPU_STEP_FUNCTION(add_get_set_array_element_uint64_t) {
    std::array<uint64_t, TEST_ARRAY_LEN> init1 = MiniSim::makeInit<uint64_t>();
    std::array<uint64_t, TEST_ARRAY_LEN> init2 = MiniSim::makeInit<uint64_t>(TEST_ARRAY_OFFSET);
    // Test Set + Get (Description set value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint64_t>("uint64_t_a_", TEST_ARRAY_LEN - 1, static_cast<uint64_t>(TEST_VALUE) * 2), init1[TEST_ARRAY_LEN - 1]);
    // Test Get (Host func set value)
    EXPECT_EQ(FLAMEGPU->environment.get<uint64_t>("uint64_t_a_", TEST_ARRAY_LEN - 1), static_cast<uint64_t>(TEST_VALUE) * 2);
    // Add
    FLAMEGPU->environment.add<uint64_t, TEST_ARRAY_LEN>("uint64_t_a", init2);
    // Test Set (Host func added value)
    EXPECT_EQ(FLAMEGPU->environment.set<uint64_t>("uint64_t_a", TEST_ARRAY_LEN - 1, static_cast<uint64_t>(TEST_VALUE)), init2[TEST_ARRAY_LEN - 1]);
    // Test Get
    EXPECT_EQ(FLAMEGPU->environment.get<uint64_t>("uint64_t_a", TEST_ARRAY_LEN - 1), static_cast<uint64_t>(TEST_VALUE));
    // Reset for next iteration
    FLAMEGPU->environment.set<uint64_t>("uint64_t_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1]);
    FLAMEGPU->environment.remove<uint64_t>("uint64_t_a");
}

FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_float) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<float, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<float>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("float_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("float_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("float_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("float_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("float_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("float_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<float>("float_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<float>("float_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_double) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<double, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<double>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("double_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("double_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("double_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("double_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("double_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("double_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<double>("double_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<double>("double_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_int8_t) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<int8_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<int8_t>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int8_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int8_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("int8_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int8_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("int8_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("int8_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int8_t>("int8_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int8_t>("int8_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_uint8_t) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<uint8_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<uint8_t>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint8_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint8_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("uint8_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint8_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("uint8_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("uint8_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint8_t>("uint8_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint8_t>("uint8_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_int16_t) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<int16_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<int16_t>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int16_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int16_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("int16_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int16_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("int16_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("int16_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int16_t>("int16_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int16_t>("int16_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_uint16_t) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<uint16_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<uint16_t>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint16_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint16_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("uint16_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint16_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("uint16_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("uint16_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint16_t>("uint16_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint16_t>("uint16_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_int32_t) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<int32_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<int32_t>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int32_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int32_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("int32_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("int32_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("int32_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("int32_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int32_t>("int32_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int32_t>("int32_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_uint32_t) {
    uint64_t _a = static_cast<uint64_t>(TEST_VALUE);
    std::array<uint32_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<uint32_t>(i);
        _b[i] = static_cast<uint64_t>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint32_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint32_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("uint32_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint32_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("uint32_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<uint64_t>("uint32_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint32_t>("uint32_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint32_t>("uint32_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_int64_t) {
    float _a = static_cast<float>(TEST_VALUE);
    std::array<int64_t, TEST_ARRAY_LEN> b;
    std::array<float, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<int64_t>(i);
        _b[i] = static_cast<float>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<float, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<float>("int64_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<float>("int64_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("int64_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<float>("int64_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<float>("int64_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<float>("int64_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int64_t>("int64_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int64_t>("int64_t_a_"));
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyType_uint64_t) {
    float _a = static_cast<float>(TEST_VALUE);
    std::array<uint64_t, TEST_ARRAY_LEN> b;
    std::array<float, TEST_ARRAY_LEN> _b;
    for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
        b[i] = static_cast<uint64_t>(i);
        _b[i] = static_cast<float>(i);
    }
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray = &HostEnvironment::set<float, TEST_ARRAY_LEN>;
    EXPECT_THROW(FLAMEGPU->environment.set<float>("uint64_t_", _a), InvalidEnvPropertyType);
    // EXPECT_THROW(FLAMEGPU->environment.set<float>("uint64_t_a_", _b), InvalidEnvPropertyType);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("uint64_t_a_", _b), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.set<float>("uint64_t_a_", 0, _a), InvalidEnvPropertyType);

    EXPECT_THROW(FLAMEGPU->environment.remove<float>("uint64_t_"), InvalidEnvPropertyType);
    EXPECT_THROW(FLAMEGPU->environment.remove<float>("uint64_t_a_"), InvalidEnvPropertyType);

    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint64_t>("uint64_t_"));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<uint64_t>("uint64_t_a_"));
}

FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_float) {
    std::array<float, TEST_ARRAY_LEN> b;
    std::array<float, 1> _b1;
    std::array<float, TEST_ARRAY_LEN + 1> _b2;
    std::array<float, TEST_ARRAY_LEN * 2> _b3;
    /**
     * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
     * They don't build on Travis with implied template args
     */
    auto setArray1 = &HostEnvironment::set<float, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<float, 1>;
    auto setArray3 = &HostEnvironment::set<float, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<float, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("float_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("float_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("float_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("float_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_double) {
    std::array<double, TEST_ARRAY_LEN> b;
    std::array<double, 1> _b1;
    std::array<double, TEST_ARRAY_LEN + 1> _b2;
    std::array<double, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<double, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<double, 1>;
    auto setArray3 = &HostEnvironment::set<double, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<double, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("double_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("double_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("double_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("double_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_int8_t) {
    std::array<int8_t, TEST_ARRAY_LEN> b;
    std::array<int8_t, 1> _b1;
    std::array<int8_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<int8_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<int8_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<int8_t, 1>;
    auto setArray3 = &HostEnvironment::set<int8_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<int8_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("int8_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("int8_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("int8_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("int8_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_uint8_t) {
    std::array<uint8_t, TEST_ARRAY_LEN> b;
    std::array<uint8_t, 1> _b1;
    std::array<uint8_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<uint8_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<uint8_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<uint8_t, 1>;
    auto setArray3 = &HostEnvironment::set<uint8_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<uint8_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("uint8_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("uint8_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("uint8_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("uint8_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_int16_t) {
    std::array<int16_t, TEST_ARRAY_LEN> b;
    std::array<int16_t, 1> _b1;
    std::array<int16_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<int16_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<int16_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<int16_t, 1>;
    auto setArray3 = &HostEnvironment::set<int16_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<int16_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("int16_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("int16_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("int16_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("int16_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_uint16_t) {
    std::array<uint16_t, TEST_ARRAY_LEN> b;
    std::array<uint16_t, 1> _b1;
    std::array<uint16_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<uint16_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<uint16_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<uint16_t, 1>;
    auto setArray3 = &HostEnvironment::set<uint16_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<uint16_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("uint16_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("uint16_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("uint16_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("uint16_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_int32_t) {
    std::array<int32_t, TEST_ARRAY_LEN> b;
    std::array<int32_t, 1> _b1;
    std::array<int32_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<int32_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<int32_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<int32_t, 1>;
    auto setArray3 = &HostEnvironment::set<int32_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<int32_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("int32_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("int32_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("int32_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("int32_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_uint32_t) {
    std::array<uint32_t, TEST_ARRAY_LEN> b;
    std::array<uint32_t, 1> _b1;
    std::array<uint32_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<uint32_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<uint32_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<uint32_t, 1>;
    auto setArray3 = &HostEnvironment::set<uint32_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<uint32_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("uint32_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("uint32_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("uint32_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("uint32_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_int64_t) {
    std::array<int64_t, TEST_ARRAY_LEN> b;
    std::array<int64_t, 1> _b1;
    std::array<int64_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<int64_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<int64_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<int64_t, 1>;
    auto setArray3 = &HostEnvironment::set<int64_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<int64_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("int64_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("int64_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("int64_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("int64_t_a_", _b3), OutOfBoundsException);
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyLength_uint64_t) {
    std::array<uint64_t, TEST_ARRAY_LEN> b;
    std::array<uint64_t, 1> _b1;
    std::array<uint64_t, TEST_ARRAY_LEN + 1> _b2;
    std::array<uint64_t, TEST_ARRAY_LEN * 2> _b3;
    /**
    * It is necessary to use function pointers for any functions that are templated with 2 args and overloaded
    * They don't build on Travis with implied template args
    */
    auto setArray1 = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN>;
    auto setArray2 = &HostEnvironment::set<uint64_t, 1>;
    auto setArray3 = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN + 1>;
    auto setArray4 = &HostEnvironment::set<uint64_t, TEST_ARRAY_LEN * 2>;
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray1)("uint64_t_a_", b));
    EXPECT_THROW((FLAMEGPU->environment.*setArray2)("uint64_t_a_", _b1), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray3)("uint64_t_a_", _b2), OutOfBoundsException);
    EXPECT_THROW((FLAMEGPU->environment.*setArray4)("uint64_t_a_", _b3), OutOfBoundsException);
}

FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_float) {
    float c = static_cast<float>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<float>("float_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<float>("float_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_double) {
    double c = static_cast<double>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<double>("double_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<double>("double_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_int8_t) {
    int8_t c = static_cast<int8_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<int8_t>("int8_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<int8_t>("int8_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_uint8_t) {
    uint8_t c = static_cast<uint8_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<uint8_t>("uint8_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<uint8_t>("uint8_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_int16_t) {
    int16_t c = static_cast<int16_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<int16_t>("int16_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<int16_t>("int16_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_uint16_t) {
    uint16_t c = static_cast<uint16_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<uint16_t>("uint16_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<uint16_t>("uint16_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_int32_t) {
    int32_t c = static_cast<int32_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<int32_t>("int32_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<int32_t>("int32_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_uint32_t) {
    uint32_t c = static_cast<uint32_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<uint32_t>("uint32_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<uint32_t>("uint32_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_int64_t) {
    int64_t c = static_cast<int64_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<int64_t>("int64_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<int64_t>("int64_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}
FLAMEGPU_STEP_FUNCTION(ExceptionPropertyRange_uint64_t) {
    uint64_t c = static_cast<uint64_t>(TEST_VALUE);
    for (int i = 0; i < 5; ++i) {
        EXPECT_THROW(FLAMEGPU->environment.set<uint64_t>("uint64_t_a_", TEST_ARRAY_LEN + i, c), OutOfBoundsException);
        EXPECT_THROW(FLAMEGPU->environment.get<uint64_t>("uint64_t_a_", TEST_ARRAY_LEN + i), OutOfBoundsException);
    }
}

FLAMEGPU_STEP_FUNCTION(ExceptionPropertyDoesntExist) {
    float a = static_cast<float>(TEST_VALUE);
    EXPECT_THROW(FLAMEGPU->environment.get<float>("a"), InvalidEnvProperty);
    FLAMEGPU->environment.add<float>("a", a);
    EXPECT_EQ(FLAMEGPU->environment.get<float>("a"), a);
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<float>("a"));
    EXPECT_THROW(FLAMEGPU->environment.get<float>("a"), InvalidEnvProperty);
    EXPECT_THROW(FLAMEGPU->environment.remove<float>("a"), InvalidEnvProperty);
    // array version
    auto addArray = &HostEnvironment::add<int, TEST_ARRAY_LEN>;
    std::array<int, TEST_ARRAY_LEN> b;
    // EXPECT_NO_THROW(FLAMEGPU->environment.add<int>("a", b));  // Doesn't build on Travis
    EXPECT_NO_THROW((FLAMEGPU->environment.*addArray)("a", b, false));
    EXPECT_NO_THROW(FLAMEGPU->environment.get<int>("a"));
    EXPECT_NO_THROW(FLAMEGPU->environment.get<int>("a", 1));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int>("a"));
    EXPECT_THROW(FLAMEGPU->environment.get<float>("a"), InvalidEnvProperty);
    EXPECT_THROW(FLAMEGPU->environment.get<float>("a", 1), InvalidEnvProperty);
    EXPECT_THROW(FLAMEGPU->environment.remove<float>("a"), InvalidEnvProperty);
}

FLAMEGPU_STEP_FUNCTION(ExceptionPropertyReadOnly) {
    float a = static_cast<float>(TEST_VALUE);
    EXPECT_NO_THROW(FLAMEGPU->environment.add<float>("a", a, true));
    EXPECT_THROW(FLAMEGPU->environment.set<float>("a", a), ReadOnlyEnvProperty);
    EXPECT_EQ(FLAMEGPU->environment.get<float>("a"), a);
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<float>("a"));
    EXPECT_THROW(FLAMEGPU->environment.set<float>("a", a), InvalidEnvProperty);
    EXPECT_NO_THROW(FLAMEGPU->environment.add<float>("a", a, false));
    EXPECT_NO_THROW(FLAMEGPU->environment.set<float>("a", a));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<float>("a"));
    // array version
    auto addArray = &HostEnvironment::add<int, TEST_ARRAY_LEN>;
    auto setArray = &HostEnvironment::set<int, TEST_ARRAY_LEN>;
    std::array<int, TEST_ARRAY_LEN> b;
    // EXPECT_NO_THROW(FLAMEGPU->environment.add<int>("a", b));  // Doesn't build on Travis
    EXPECT_NO_THROW((FLAMEGPU->environment.*addArray)("a", b, true));
    // EXPECT_THROW(FLAMEGPU->environment.set<int>("a", b), ReadOnlyEnvProperty);  // Doesn't build on Travis
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("a", b), ReadOnlyEnvProperty);
    EXPECT_NO_THROW(FLAMEGPU->environment.get<int>("a"));
    EXPECT_NO_THROW(FLAMEGPU->environment.get<int>("a", 1));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int>("a"));
    EXPECT_THROW((FLAMEGPU->environment.*setArray)("a", b), InvalidEnvProperty);
    EXPECT_NO_THROW((FLAMEGPU->environment.*addArray)("a", b, false));
    EXPECT_NO_THROW((FLAMEGPU->environment.*setArray)("a", b));
    EXPECT_NO_THROW(FLAMEGPU->environment.remove<int>("a"));
}

FLAMEGPU_STEP_FUNCTION(BoolWorks) {
    {
        bool a = true;
        bool b = false;
        FLAMEGPU->environment.add<bool>("a", a);
        EXPECT_EQ(FLAMEGPU->environment.set<bool>("a", b), a);
        EXPECT_EQ(FLAMEGPU->environment.get<bool>("a"), b);
        EXPECT_NO_THROW(FLAMEGPU->environment.remove<bool>("a"));
    }
    // array version
    {
        std::array<bool, TEST_ARRAY_LEN> a;
        std::array<bool, TEST_ARRAY_LEN> b;
        std::array<bool, TEST_ARRAY_LEN> res;
        for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
            a[i] = true;
            b[i] = false;
        }
        FLAMEGPU->environment.add<bool, TEST_ARRAY_LEN>("a", a);
        res = FLAMEGPU->environment.set<bool, TEST_ARRAY_LEN>("a", b);
        for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
            EXPECT_EQ(res[i], a[i]);
        }
        res = FLAMEGPU->environment.get<bool, TEST_ARRAY_LEN>("a");
        for (int i = 0; i < TEST_ARRAY_LEN; ++i) {
            EXPECT_EQ(res[i], b[i]);
        }
        EXPECT_NO_THROW(FLAMEGPU->environment.remove<bool>("a"));
    }
}

/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class HostEnvironmentTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};
}  // namespace

TEST_F(HostEnvironmentTest, AddGet_SetGetfloat) {
    ms->model.addStepFunction(add_get_set_float);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetdouble) {
    ms->model.addStepFunction(add_get_set_double);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetint8_t) {
    ms->model.addStepFunction(add_get_set_int8_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetuint8_t) {
    ms->model.addStepFunction(add_get_set_uint8_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetint16_t) {
    ms->model.addStepFunction(add_get_set_int16_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetuint16_t) {
    ms->model.addStepFunction(add_get_set_uint16_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetint32_t) {
    ms->model.addStepFunction(add_get_set_int32_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetuint32_t) {
    ms->model.addStepFunction(add_get_set_uint32_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetint64_t) {
    ms->model.addStepFunction(add_get_set_int64_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetuint64_t) {
    ms->model.addStepFunction(add_get_set_uint64_t);
    // Test Something
    ms->run();
}

TEST_F(HostEnvironmentTest, AddGet_SetGetarray_float) {
    ms->model.addStepFunction(add_get_set_array_float);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_double) {
    ms->model.addStepFunction(add_get_set_array_double);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_int8_t) {
    ms->model.addStepFunction(add_get_set_array_int8_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_uint8_t) {
    ms->model.addStepFunction(add_get_set_array_uint8_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_int16_t) {
    ms->model.addStepFunction(add_get_set_array_int16_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_uint16_t) {
    ms->model.addStepFunction(add_get_set_array_uint16_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_int32_t) {
    ms->model.addStepFunction(add_get_set_array_int32_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_uint32_t) {
    ms->model.addStepFunction(add_get_set_array_uint32_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_int64_t) {
    ms->model.addStepFunction(add_get_set_array_int64_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_uint64_t) {
    ms->model.addStepFunction(add_get_set_array_uint64_t);
    // Test Something
    ms->run();
}

TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_float) {
    ms->model.addStepFunction(add_get_set_array_element_float);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_double) {
    ms->model.addStepFunction(add_get_set_array_element_double);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_int8_t) {
    ms->model.addStepFunction(add_get_set_array_element_int8_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_uint8_t) {
    ms->model.addStepFunction(add_get_set_array_element_uint8_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_int16_t) {
    ms->model.addStepFunction(add_get_set_array_element_int16_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_uint16_t) {
    ms->model.addStepFunction(add_get_set_array_element_uint16_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_int32_t) {
    ms->model.addStepFunction(add_get_set_array_element_int32_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_uint32_t) {
    ms->model.addStepFunction(add_get_set_array_element_uint32_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_int64_t) {
    ms->model.addStepFunction(add_get_set_array_element_int64_t);
    // Test Something
    ms->run();
}
TEST_F(HostEnvironmentTest, AddGet_SetGetarray_element_uint64_t) {
    ms->model.addStepFunction(add_get_set_array_element_uint64_t);
    // Test Something
    ms->run();
}

// ExceptionPropertyType_float
TEST_F(HostEnvironmentTest, ExceptionPropertyType_float) {
    ms->model.addStepFunction(ExceptionPropertyType_float);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_double) {
    ms->model.addStepFunction(ExceptionPropertyType_double);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_int8_t) {
    ms->model.addStepFunction(ExceptionPropertyType_int8_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_uint8_t) {
    ms->model.addStepFunction(ExceptionPropertyType_uint8_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_int16_t) {
    ms->model.addStepFunction(ExceptionPropertyType_int16_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_uint16_t) {
    ms->model.addStepFunction(ExceptionPropertyType_uint16_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_int32_t) {
    ms->model.addStepFunction(ExceptionPropertyType_int32_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_uint32_t) {
    ms->model.addStepFunction(ExceptionPropertyType_uint32_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_int64_t) {
    ms->model.addStepFunction(ExceptionPropertyType_int64_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyType_uint64_t) {
    ms->model.addStepFunction(ExceptionPropertyType_uint64_t);
    // Test Something
    ms->run(1);
}

// ExceptionPropertyLength_float
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_float) {
    ms->model.addStepFunction(ExceptionPropertyLength_float);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_double) {
    ms->model.addStepFunction(ExceptionPropertyLength_double);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_int8_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_int8_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_uint8_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_uint8_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_int16_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_int16_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_uint16_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_uint16_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_int32_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_int32_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_uint32_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_uint32_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_int64_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_int64_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyLength_uint64_t) {
    ms->model.addStepFunction(ExceptionPropertyLength_uint64_t);
    // Test Something
    ms->run(1);
}

// ExceptionPropertyRange_float
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_float) {
    ms->model.addStepFunction(ExceptionPropertyRange_float);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_double) {
    ms->model.addStepFunction(ExceptionPropertyRange_double);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_int8_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_int8_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_uint8_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_uint8_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_int16_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_int16_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_uint16_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_uint16_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_int32_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_int32_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_uint32_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_uint32_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_int64_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_int64_t);
    // Test Something
    ms->run(1);
}
TEST_F(HostEnvironmentTest, ExceptionPropertyRange_uint64_t) {
    ms->model.addStepFunction(ExceptionPropertyRange_uint64_t);
    // Test Something
    ms->run(1);
}

// ExceptionPropertyDoesntExist
TEST_F(HostEnvironmentTest, ExceptionPropertyDoesntExist) {
    ms->model.addStepFunction(ExceptionPropertyDoesntExist);
    // Test Something
    ms->run(1);
}

// ExceptionPropertyReadOnly
TEST_F(HostEnvironmentTest, ExceptionPropertyReadOnly) {
    ms->model.addStepFunction(ExceptionPropertyReadOnly);
    // Test Something
    ms->run(1);
}

// bool
TEST_F(HostEnvironmentTest, BoolWorks) {
    ms->model.addStepFunction(BoolWorks);
    // Test Something
    ms->run(1);
}
