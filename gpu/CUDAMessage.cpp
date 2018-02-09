/**
* @file CUDAMessage.cpp
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDAMessage.h"
#include "CUDAMessageList.h"
#include "CUDAErrorChecking.h"

#include "../model/MessageDescription.h"
#include "../model/AgentFunctionDescription.h"
#include "../runtime/cuRVE/curve.h"


/**
* CUDAMessage class
* @brief allocates the hash table/list for message variables and copy the list to device
*/
CUDAMessage::CUDAMessage(const MessageDescription& description) : message_description(description), max_list_size(0)
{

}


/**
 * A destructor.
 * @brief Destroys the CUDAMessage object
 */
CUDAMessage::~CUDAMessage(void)
{

}


/**
* @brief Returns message description
* @param none
* @return MessageDescription object
*/
const MessageDescription& CUDAMessage::getMessageDescription() const
{
    return message_description;
}

/**
* @brief Returns the maximum list size
* @param none
* @return maximum size list that is equal to the maxmimum population size
*/
unsigned int CUDAMessage::getMaximumListSize() const
{
    return max_list_size;
}

/**
@bug message_name is input or output, run some tests to see which one is correct
*/
void CUDAMessage::mapRuntimeVariables(const AgentFunctionDescription& func) const
{

    const std::string message_name = message_description.getName();
    //loop through the agents variables to map each variable name using cuRVE
    for (VariableMapPair mmp : message_description.getVariableMap())
    {
        //get a device pointer for the agent variable name
        void* d_ptr = message_list->getMessageListVariablePointer(mmp.first);

        //map using curve
		CurveVariableHash var_hash = curveVariableRuntimeHash(mmp.first.c_str());
		CurveVariableHash message_hash = curveVariableRuntimeHash(message_name.c_str());
		CurveVariableHash func_hash = curveVariableRuntimeHash(func.getName().c_str());

        // get the agent variable size
        size_t size;
        size = message_description.getMessageVariableSize(mmp.first.c_str());

       // maximum population num
        unsigned int length = this->getMaximumListSize();

		curveRegisterVariableByHash(var_hash + message_hash + func_hash, d_ptr, size, length);
    }

}

void CUDAMessage::unmapRuntimeVariables(const AgentFunctionDescription& func) const
{

    const std::string message_name = message_description.getName();
    //loop through the agents variables to map each variable name using cuRVE
    for (VariableMapPair mmp : message_description.getVariableMap())
    {
        //get a device pointer for the agent variable name
        void* d_ptr = message_list->getMessageListVariablePointer(mmp.first);

        //unmap using curve
		CurveVariableHash var_hash = curveVariableRuntimeHash(mmp.first.c_str());
		CurveVariableHash message_hash = curveVariableRuntimeHash(message_name.c_str());
		CurveVariableHash func_hash = curveVariableRuntimeHash(func.getName().c_str());

		curveUnregisterVariableByHash(var_hash + message_hash + func_hash);
    }

}

