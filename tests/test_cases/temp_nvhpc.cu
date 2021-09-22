#include "flamegpu/flamegpu.h"
#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_nvhpc {

FLAMEGPU_AGENT_FUNCTION(cudacxx_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

const char* rtc_test_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}
)###";

TEST(testNVHPC, RTCElapsedTime) {
    ModelDescription m("m");
    AgentDescription &agent = m.newAgent("agent");

    // Using newRTCFunction and newFunction in the same compilation unit appears to cause the segfault within newRTCFunction.
    // Comment out either call to remove the segfault.
    agent.newFunction("cudacxx_test_func", cudacxx_test_func);
    AgentFunctionDescription &func = agent.newRTCFunction("rtc_test_func", rtc_test_func);
}

}  // namespace test_nvhpc
}  // namespace tests
}  // namespace flamegpu