/**
* @file CUDAMessage.h
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#ifndef CUDAMESSAGE_H_
#define CUDAMESSAGE_H_


//forward declare classes from other modules
class MessageDescription;

class CUDAMessage
{
public:
    CUDAMessage(const MessageDescription& description);
    virtual ~CUDAMessage(void);

    const MessageDescription& getMessageDescription() const;
    unsigned int getMaximumListSize() const;

protected:


private:
    const MessageDescription& message_description;
    unsigned int max_list_size;

};

#endif
