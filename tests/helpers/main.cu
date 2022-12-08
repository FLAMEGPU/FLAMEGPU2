#include <cuda_runtime.h>
#include <cstdio>
#include <map>

#include "flamegpu/gpu/CUDASimulation.h"
#include "gtest/gtest.h"
#include "helpers/device_initialisation.h"
#include "flamegpu/io/Telemetry.h"


GTEST_API_ int main(int argc, char **argv) {
    // Get the current status of telemetry, to control if test suite results shold be submit or not
    const bool telemetryEnabled = flamegpu::io::Telemetry::isEnabled();
    // Disable telemetry for simulation / ensemble objects in the test suite.
    flamegpu::io::Telemetry::disable();
    // Suppress the notice about telemetry.
    flamegpu::io::Telemetry::suppressNotice();
    // Time the cuda agent model initialisation, to check it creates the context.
    flamegpu::tests::timeCUDASimulationContextCreationTest();
    // google test doesnt have a verbose mode so check for verbosity flag for telemetry output
    bool verbose = false;
    for (int i = 0; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if ((arg.compare("-v") == 0) || (arg.compare("--verbose") == 0))
            verbose = true;
    }
    // Run the main google test body
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    auto rtn = RUN_ALL_TESTS();
    // send telemetry if it was enabled globally (dont bother nagging on test suite)
    // Also only share if more than one test was ran (i.e. don't spam for ctest when using google test discover.)
    // if (telemetryEnabled && ::testing::UnitTest::GetInstance()->test_to_run_count() > 1) {
    //     printf("testing\n");
    //     return 1;
    //     std::map<std::string, std::string> telemetry_payload;
    //     telemetry_payload["TestOutcome"] = rtn ? "Passed" : "Failed";
    //     telemetry_payload["TestsPassed"] = std::to_string(::testing::UnitTest::GetInstance()->successful_test_count());
    //     telemetry_payload["TestsToRun"] = std::to_string(::testing::UnitTest::GetInstance()->test_to_run_count());
    //     telemetry_payload["TestsTotal"] = std::to_string(::testing::UnitTest::GetInstance()->total_test_count());
    //     // generate telemetry data
    //     std::string telemetry_data = flamegpu::io::Telemetry::generateData("googletest-run", telemetry_payload);
    //     // send telemetry
    //     flamegpu::io::Telemetry::sendData(telemetry_data);
    //     // print telemetry
    //     if (verbose) {
    //         fprintf(stdout, "Telemetry packet sent to '%s' json was: %s\n", flamegpu::io::Telemetry::TELEMETRY_ENDPOINT, telemetry_data.c_str());
    //     }
    // }
    // Reset all cuda devices for memcheck / profiling purposes.
    int devices = 0;
    gpuErrchk(cudaGetDeviceCount(&devices));
    if (devices > 0) {
        for (int device = 0; device < devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaDeviceReset());
        }
    }

    return rtn;
}
