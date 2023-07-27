#ifndef TESTS_HELPERS_DEVICE_INITIALISATION_H_
#define TESTS_HELPERS_DEVICE_INITIALISATION_H_

namespace flamegpu {
namespace tests {
/**
 *  Test that no cuda context is established prior to  CUDASimulation::applyConfig_derived().
 *
 * I.e. make sure that nothing occurs before device selection.
 *
 * This is performed by checking the cuda driver api current context is null, rather than the runtime api + timing based approach which was fluffy / driver updates improving context creation time would break the test.
 *
 * @note - This needs to be called first, and only once, hence it is not a true google test.
 * @todo - It may be better places in it's own test binary, orchestrated via ctest to ensure it is first and only called once.
*/
void runCUDASimulationContextCreationTest();

/**
 * Determine the success of the cuda context creation test.
 * @returns bool indicator of the test passing
 */
bool getCUDASimulationContextCreationTestResult();

}  // namespace tests
}  // namespace flamegpu

#endif  // TESTS_HELPERS_DEVICE_INITIALISATION_H_
