/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_all.cpp
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#define BOOST_TEST_MODULE "All Unit Tests for FLAMEGPU 2"

/*
//Debug new wrapper to test for memory leaks
#ifdef _DEBUG
#define DEBUG_NEW new( _CLIENT_BLOCK, __FILE__, __LINE__)
#else
#define DEBUG_CLIENTBLOCK
#endif // _DEBUG
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
*/


#include <boost/test/included/unit_test.hpp>

#include "test_cases/model/test_model_validation.h"
#include "test_cases/model/test_message_validation.h"
#include "test_cases/pop/test_pop_validation.h"
#include "test_cases/sim/test_sim_validation.h"
#include "test_cases/gpu/test_gpu_validation.h"

// Disabled, currently breaks compilation.
// #include "test_cases/runtime/test_func_pointer.h"
