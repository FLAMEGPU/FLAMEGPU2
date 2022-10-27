#include <cuda_runtime.h>
#include <cstdio>

#include "flamegpu/gpu/CUDASimulation.h"
#include "gtest/gtest.h"
#include "helpers/device_initialisation.h"


GTEST_API_ int main(int argc, char **argv) {
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

  return rtn;
}
