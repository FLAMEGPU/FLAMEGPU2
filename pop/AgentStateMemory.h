 /**
 * @file AgentStateMemory.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef AGENTSTATEDESCRIPTION_H_
#define AGENTSTATEDESCRIPTION_H_


#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <memory>
#include <boost/any.hpp>
#include <typeinfo>


#include "../model/AgentDescription.h"

typedef std::map<const std::string, std::unique_ptr<std::vector<boost::any>>> StateMemoryMap;
typedef std::pair<const std::string, std::unique_ptr<std::vector<boost::any>>> StateMemoryMapPair;

class AgentStateMemory  // agent_list
{
public:
    AgentStateMemory(const AgentDescription &description, const std::string agent_state) ;
    virtual ~AgentStateMemory() {}

    unsigned int getSize() const;

    //void init(const AgentDescription &agent_description);
    unsigned int creatNewInstance();

    std::vector<boost::any>& getMemoryVector(const std::string variable_name);

    const std::vector<boost::any>& getReadOnlyMemoryVector(const std::string variable_name) const;

    //todo:templated get vector function with boost any cast

    const std::type_info& getVariableType(std::string variable_name); //const

    bool isSameDescription(const AgentDescription& description) const;

protected:
    const AgentDescription &agent_description;
    const std::string agent_state;
    StateMemoryMap state_memory;
    unsigned int size;
};

#endif /* AGENTSTATEDESCRIPTION_H_ */
