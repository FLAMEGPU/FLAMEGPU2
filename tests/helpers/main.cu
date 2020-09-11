#include <cuda_runtime.h>
#include <cstdio>

#include "flamegpu/gpu/CUDASimulation.h"
#include "gtest/gtest.h"
#include "helpers/device_initialisation.h"


GTEST_API_ int main(int argc, char **argv) {
  // Disable auto reset, as it slows down test execution quite a bit.
  CUDASimulation::AUTO_CUDA_DEVICE_RESET = false;
  // Time the cuda agent model initialisation, to check it creates the context.
  timeCUDASimulationContextCreationTest();
  // Run the main google test body
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  auto rtn = RUN_ALL_TESTS();
  cudaDeviceReset();
  return rtn;
}
