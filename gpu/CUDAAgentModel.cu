/**
 * @file CUDAAgentModel.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "CUDAAgentModel.h"

#include "../model/ModelDescription.h"
#include "../pop/AgentPopulation.h"
#include "../sim/Simulation.h"

// include FLAMEGPU kernel wrapper
#include "../runtime/agent_function.cu"

// agent_map is a type CUDAAgentMap
/**
* CUDAAgentModel class
* @brief populates CUDA agent map, CUDA message map
*/
CUDAAgentModel::CUDAAgentModel(const ModelDescription& description) : model_description(description), agent_map(), curve(cuRVEInstance::getInstance()) //, message_map(), function_map() {
{
    //create a reference to curve to ensure that it is initialised. This is a singleton class so will only be done once regardless of the number of CUDAgentModels.

    //populate the CUDA agent map
    const AgentMap &am = model_description.getAgentMap();
    AgentMap::const_iterator it; // const_iterator returns a reference to a constant value (const T&) and prevents modification of the reference value

    //create new cuda agent and add to the map
    for(it = am.begin(); it != am.end(); it++)
    {
        agent_map.insert(CUDAAgentMap::value_type(it->first, std::unique_ptr<CUDAAgent>(new CUDAAgent(it->second))));
    } // insert into map using value_type

    //same for messages - Todo
    //same for functions - Todo

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

/**
 * A destructor.
 * @brief Destroys the CUDAAgentModel object
 */
CUDAAgentModel::~CUDAAgentModel()
{
    //unique pointers cleanup by automatically
}

/**
* @brief Sets the initial population data
* @param AgentPopulation object
* @return none
*/
void CUDAAgentModel::setInitialPopulationData(AgentPopulation& population)
{
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end())
    {
        //throw std::runtime_error("CUDA agent not found. This should not happen.");
        throw InvalidCudaAgent();
    }

    //create agent state lists
    it->second->setInitialPopulationData(population);
}

/**
* @brief Sets the population data
* @param AgentPopulation object
* @return none
*/
void CUDAAgentModel::setPopulationData(AgentPopulation& population)
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

void CUDAAgentModel::getPopulationData(AgentPopulation& population)
{
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end())
    {
        throw InvalidCudaAgent("CUDA agent not found.");
    }

    //create agent state lists
    it->second->getPopulationData(population);
}


void CUDAAgentModel::addSimulation(const Simulation& simulation) {}

void CUDAAgentModel::step(const Simulation& simulation)
{

    //for each each sim layer
    for (unsigned int i = 0; i < simulation.getLayerCount(); i++)
    {
        const FunctionDescriptionVector& functions = simulation.getFunctionsAtLayer(i);

        //for each func function
        for (AgentFunctionDescription func_des : functions)
        {

            std::string agent_name = func_des.getParent().getName();
            const CUDAAgent& cuda_agent = getCUDAAgent(func_des.getParent().getName());

            // set namespace
            cuda_agent.setNamespace(agent_name);


            //configure runtime access of the functions variables within the FLAME_API object
            cuda_agent.mapRuntimeVariables(func_des);

            //get the agent function
            FLAMEGPU_AGENT_FUNCTION_POINTER* agent_func = func_des.getFunction();

            //host_pointer
            FLAMEGPU_AGENT_FUNCTION_POINTER h_func_ptr;

            cudaMemcpyFromSymbol(&h_func_ptr, *agent_func, sizeof(FLAMEGPU_AGENT_FUNCTION_POINTER));

            //call the agent function wrapper which creates an instance of FLAMEGPU_API on the device to pass to the agent function.
            //TODO: Kernel dimensions will come from the CUDAAgent state list size
            //calculate the grid block size for main agent function

            int blockSize; // The launch configurator returned block size
            int minGridSize; // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize; // The actual grid size needed, based on input size

            int state_list_size = cuda_agent.getMaximumListSize();

            //calculate the grid block size for main agent function
            cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, agent_function_wrapper, 0, state_list_size);
            //cudaOccupancyMaxPotentialBlockSizeVariableSMem ?

            // Round up according to CUDAAgent state list size
            gridSize = (state_list_size + blockSize - 1) / blockSize;

			//hash agent name
			CurveNamespaceHash agentname_hash = curveVariableRuntimeHash(agent_name.c_str());


			agent_function_wrapper << <gridSize, blockSize >> >(agentname_hash, h_func_ptr, state_list_size);
           // agent_function_wrapper <<<1,256>>>(h_func_ptr, state_list_size);
            cudaDeviceSynchronize();

            //unmap the function variables
            cuda_agent.unmapRuntimeVariables(func_des);
        }
    }
}

/**
* @brief initialize CUDA params (e.g: set CUDA device)
* @warning not tested
*/
void CUDAAgentModel::init(void)   // (int argc, char** argv)
{

    cudaError_t cudaStatus;
    int device;
    int device_count;

    //default device
    device = 0;
    cudaStatus = cudaGetDeviceCount(&device_count);

    if (cudaStatus != cudaSuccess)
    {
        throw InvalidCUDAdevice("Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?");
        exit(0);
    }
    if (device_count == 0)
    {
        throw InvalidCUDAdevice("Error no CUDA devices found!");
        exit(0);
    }

    // Select device
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess)
    {
        throw InvalidCUDAdevice("Error setting CUDA device!");
        exit(0);
    }
}


/**
* @brief simulates functions
* @param   object
* @return none
* @todo not yet completed
* @warning not tested
*/
void CUDAAgentModel::simulate(const Simulation& sim)
{
    if (agent_map.size() == 0)
        //throw std::runtime_error("CUDA agent map size is zero"); // population size = 0 ? do we mean checking the number of elements in the map container?
        throw InvalidCudaAgentMapSize();


    //CUDAAgentMap::iterator it;


    //check any CUDAAgents with population size == 0
    //if they have executable functions then these can be ignored
    //if they have agent creations then buffer space must be allocated for them
}

const CUDAAgent& CUDAAgentModel::getCUDAAgent(std::string agent_name) const
{
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end())
    {
        throw InvalidCudaAgent("CUDA agent not found.");
    }

    return *(it->second);
}

