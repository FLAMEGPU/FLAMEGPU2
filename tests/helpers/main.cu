#include <cuda_runtime.h>
#include <cstdio>
#include <map>

#include "flamegpu/simulation/CUDASimulation.h"
#include "gtest/gtest.h"
#include "helpers/device_initialisation.h"
#include "flamegpu/io/Telemetry.h"
#include "flamegpu/detail/TestSuiteTelemetry.h"


GTEST_API_ int main(int argc, char **argv) {
    // Get the current status of telemetry, to control if test suite results shold be submit or not
    const bool telemetryEnabled = flamegpu::io::Telemetry::isEnabled();
    // Disable telemetry for simulation / ensemble objects in the test suite.
    flamegpu::io::Telemetry::disable();
    // Suppress the notice about telemetry.
    flamegpu::io::Telemetry::suppressNotice();
    // Time the cuda agent model initialisation, to check it creates the context.
    flamegpu::tests::timeCUDASimulationContextCreationTest();
    // Run the main google test body
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    auto rtn = RUN_ALL_TESTS();
    // Reset all cuda devices for memcheck / profiling purposes.
    int devices = 0;
    gpuErrchk(cudaGetDeviceCount(&devices));
    if (devices > 0) {
        for (int device = 0; device < devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaDeviceReset());
        }
    }
    // If there were more than 1 tests selected, (to exlcude bad filters and FLAMEGPU_USE_GTEST_DISCOVER related spam)
    if (telemetryEnabled && ::testing::UnitTest::GetInstance()->test_to_run_count() > 1) {
        // Detect if -v / --verbose was passed to the test suite binary to log the payload command
        bool verbose = false;
        for (int i = 0; i < argc; i++) {
            std::string arg = std::string(argv[i]);
            if ((arg.compare("-v") == 0) || (arg.compare("--verbose") == 0))
                verbose = true;
        }
        // re-enable telemetry so this will actually send.
        flamegpu::io::Telemetry::enable();
        // Submit the test results, do not handle the result and silently fail if needed
        std::string outcome = rtn == EXIT_SUCCESS ? "Passed" : "Failed";
        flamegpu::detail::TestSuiteTelemetry::sendResults("googletest-run"
            , outcome
            , ::testing::UnitTest::GetInstance()->total_test_count()
            , ::testing::UnitTest::GetInstance()->test_to_run_count()
            , ::testing::UnitTest::GetInstance()->disabled_test_count()
            , ::testing::UnitTest::GetInstance()->successful_test_count()
            , ::testing::UnitTest::GetInstance()->failed_test_count()
            , verbose);
    }
    return rtn;
}
