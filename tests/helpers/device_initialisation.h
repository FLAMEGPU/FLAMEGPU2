#ifndef TESTS_HELPERS_DEVICE_INITIALISATION_H_
#define TESTS_HELPERS_DEVICE_INITIALISATION_H_


/**
 * Function to time the creation of the cuda context within the scope of FLAME GPU initialisation.
 * Must be run first to ensure a fresh context is being created, rather than within the google test suite which may be executed in a random order.
 */
void timeCUDAAgentModelContextCreationTest();

/**
 * Determine the success of the cuda context creation test.
 * @returns bool indicator of the test passing
 */
bool getCUDAAgentModelContextCreationTestResult();

#endif  // TESTS_HELPERS_DEVICE_INITIALISATION_H_
