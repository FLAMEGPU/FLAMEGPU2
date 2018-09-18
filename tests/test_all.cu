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

#include "test_model_validation.h"
#include "test_message_validation.h"
#include "test_pop_validation.h"
#include "test_sim_validation.h" 
#include "test_gpu_validation.h" 





