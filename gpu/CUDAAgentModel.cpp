/*
 * CUDAAgentModel.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 *  Last modified : 29 Nov 2016
 */

#include "CUDAAgentModel.h"

// agent_map is a type CUDAAgentMap

CUDAAgentModel::CUDAAgentModel(const ModelDescription& description) : model_description(description), agent_map()  //, message_map(), function_map() {
{

    //populate the CUDA agent map
    const AgentMap &am = model_description.getAgentMap();
    AgentMap::const_iterator it; // const_iterator returns a reference to a constant value (const T&) and prevents modification of the reference value

    //create new cuda agent and add to the map
    for(it = am.begin(); it != am.end(); it++)
    {
        agent_map.insert(CUDAAgentMap::value_type(it->first, std::unique_ptr<CUDAAgent>(new CUDAAgent(it->second))));
    } // insert into map using value_type

    //same for messages - done
    //same for functions - done

    /*Moz
    //populate the CUDA message map
    const MessageMap &mm = model_description.getMessageMap();
    MessageMap::const_iterator it;

    //create new cuda message and add to the map
    for(it = mm.begin(); it != mm.end(); it++){
    	MessageMap.insert(CUDAMessageMap::value_type(it->first, std::unique_ptr<CUDAMessage>(new CUDAMessage(it->second))));
    }


    //populate the CUDA function map
    const FunctionMap &mm = model_description.getFunctionMap();
    FunctioneMap::const_iterator it;

    for(it = mm.begin(); it != mm.end(); it++){
    	FunctionMap.insert(CUDAFunctionMap::value_type(it->first, std::unique_ptr<CUDAAgentFunction>(new CUDAAgentFunction(it->second))));
    }
    */

}

CUDAAgentModel::~CUDAAgentModel()
{
    //unique pointers cleanup by automatically
}

void CUDAAgentModel::setPopulationData(AgentPopulation& population, bool overwite_exiting)
{
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end())
    {
        //throw std::runtime_error("CUDA agent not found. This should not happen.");
        throw InvalidCudaAgent();
    }

    //create agent state lists
    it->second->setPopulationData(population);
}

void CUDAAgentModel::simulate(const Simulation& sim)  // Moz:
{
    if (agent_map.size() == 0)
        //throw std::runtime_error("CUDA agent map size is zero"); // population size = 0 ? do we mean checking the number of elements in the map container?
        throw InvalidCudaAgentMapSize();

    //CUDAAgentMap::iterator it;


    //check any CUDAAgents with population size == 0  // Moz : not sure what this means ! population size is set by default
    //if they have executable functions then these can be ignored
    //if they have agent creations then buffer space must be allocated for them
}

