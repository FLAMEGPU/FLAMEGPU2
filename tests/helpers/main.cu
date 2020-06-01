#include <cstdio>
#include <cuda_runtime.h>
#include "gtest/gtest.h"
#include "helpers/device_initialisation.h"


GTEST_API_ int main(int argc, char **argv) {
  // Time the cuda agent model initialisation, to check it creates the context.
  timeCUDAAgentModelContextCreationTest();
  // Run the main google test body
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  auto rtn = RUN_ALL_TESTS();
  cudaDeviceReset();
  return rtn;
}
