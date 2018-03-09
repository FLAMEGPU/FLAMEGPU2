 /**
 * @file CUDAMessageList.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#ifndef CUDAMESSAGELIST_H_
#define CUDAMESSAGELIST_H_

#include <string>
#include <memory>
#include <vector>
#include <map>

class CUDAMessage;
//class AgentStateMemory;

//#define UNIFIED_GPU_MEMORY

typedef std::map <std::string, void*> CUDAMemoryMap;
typedef std::pair <std::string, void*> CUDAMemoryMapPair;

class CUDAMessageList {
public:
	CUDAMessageList(CUDAMessage& cuda_message);
	virtual ~CUDAMessageList();

	void cleanupAllocatedData();

	void* getMessageListVariablePointer(std::string variable_name);

	void zeroMessageData();

protected:


	void allocateDeviceMessageList(CUDAMemoryMap &Message_list);

	void releaseDeviceMessageList(CUDAMemoryMap &Message_list);

	void zeroDeviceMessageList(CUDAMemoryMap &Message_list);


private:
	CUDAMemoryMap d_list;
	CUDAMemoryMap d_swap_list; // may not need this later
	CUDAMemoryMap d_new_list;

	unsigned int current_list_size; //???

	const CUDAMessage& message;
};

#endif
