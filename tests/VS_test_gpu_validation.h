bool gpu_test_1();
bool gpu_test_1();


BOOST_AUTO_TEST_SUITE(GPUTests)

BOOST_AUTO_TEST_CASE(GPUMemoryTest)
{
BOOST_TEST_MESSAGE( "\nTesting values copied back from device without simulating any functions .." );
    BOOST_CHECK(gpu_test_1==true);
}

BOOST_AUTO_TEST_CASE(GPUSimulationTest)
{
    BOOST_TEST_MESSAGE( "\nTesting values copied back from device after simulating functions .." );
    BOOST_CHECK(gpu_test_2==true);
}

BOOST_AUTO_TEST_SUITE_END()

