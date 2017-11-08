bool sim_test_1();
bool sim_test_2();

BOOST_AUTO_TEST_SUITE(SimTest)

BOOST_AUTO_TEST_CASE(SimulationFunctionCheck)
{
        BOOST_TEST_MESSAGE( "\nTesting simulation of functions per layers .." );
    BOOST_CHECK(sim_test_1()==true);
}

BOOST_AUTO_TEST_CASE(SimulationFunctionTypeCheck)
{
    BOOST_TEST_MESSAGE( "\nTesting variable type mismatch during the simulation .." );
    BOOST_CHECK(sim_test_2()==true);
}

BOOST_AUTO_TEST_SUITE_END()

