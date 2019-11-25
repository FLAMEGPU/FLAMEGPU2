#include <cuda_runtime.h>
#include <cstdio>
#include "gtest/gtest.h"


GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  auto rtn = RUN_ALL_TESTS();
  cudaDeviceReset();
  return rtn;
}
