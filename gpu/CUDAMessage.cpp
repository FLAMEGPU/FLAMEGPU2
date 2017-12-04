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
#include "CUDAErrorChecking.h"

#include "../model/MessageDescription.h"
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
