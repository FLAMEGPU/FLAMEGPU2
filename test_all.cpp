#define BOOST_TEST_MODULE "All Unit Tests for FLAMEGPU 2"

//requires that boost is built with the following
//bjam --toolset=msvc-12.0 address-model=64 --build-type=complete
//can optinally add --stagedir=lib\x64 stage

#include <boost/test/included/unit_test.hpp>


#include "test_model_validation.h"
#include "test_pop_validation.h"
#include "test_sim_validation.h"
//#include "test_gpu_validation.h"

