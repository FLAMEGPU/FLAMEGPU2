/**
 * @file CUDAAgentModel.cu
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "flamegpu/gpu/CUDAAgentModel.h"

#include <iostream>
#include <algorithm>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/sim/Simulation.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"

// include FLAMEGPU kernel wrapper
#include "flamegpu/runtime/agent_function.h"

/**
* CUDAAgentModel class
* @brief populates CUDA agent map, CUDA message map
*/
CUDAAgentModel::CUDAAgentModel(const ModelDescription& description)
    : model_description(description),
    agent_map(),
    curve(cuRVEInstance::getInstance()),
    message_map(),
    host_api(*this) {  // , function_map() {
    // create a reference to curve to ensure that it is initialised. This is a singleton class so will only be done once regardless of the number of CUDAgentModels.

    // populate the CUDA agent map
    const AgentMap &am = model_description.getAgentMap();
    AgentMap::const_iterator it;  // const_iterator returns a reference to a constant value (const T&) and prevents modification of the reference value

    // create new cuda agent and add to the map
    for (it = am.begin(); it != am.end(); it++) {
        agent_map.insert(CUDAAgentMap::value_type(it->first, std::unique_ptr<CUDAAgent>(new CUDAAgent(it->second))));
    }  // insert into map using value_type

    // populate the CUDA message map
    const MessageMap &mm = model_description.getMessageMap();
    MessageMap::const_iterator it_m;

    // create new cuda message and add to the map
    for (it_m = mm.begin(); it_m != mm.end(); it_m++) {
        message_map.insert(CUDAMessageMap::value_type(it_m->first, std::unique_ptr<CUDAMessage>(new CUDAMessage(it_m->second))));
    }

    /*
        // populate the CUDA function map
        const FunctionMap &mm = model_description.getFunctionMap();
        FunctioneMap::const_iterator it;

        for (it = mm.begin(); it != mm.end(); it++) {
            FunctionMap.insert(CUDAFunctionMap::value_type(it->first, std::unique_ptr<CUDAAgentFunction>(new CUDAAgentFunction(it->second))));
        }
        */
}

/**
 * A destructor.
 * @brief Destroys the CUDAAgentModel object
 */
CUDAAgentModel::~CUDAAgentModel() {
    // unique pointers cleanup by automatically
}

/**
* @brief Sets the initial population data
* @param AgentPopulation object
* @return none
*/
void CUDAAgentModel::setInitialPopulationData(AgentPopulation& population) {
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        throw InvalidCudaAgent();
    }

    /*! create agent state lists */
    it->second->setInitialPopulationData(population);
}

/**
* @brief Sets the population data
* @param AgentPopulation object
* @return none
*/
void CUDAAgentModel::setPopulationData(AgentPopulation& population) {
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        throw InvalidCudaAgent();
    }

    /*! create agent state lists */
    it->second->setPopulationData(population);
}

void CUDAAgentModel::getPopulationData(AgentPopulation& population) {
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        throw InvalidCudaAgent("CUDA agent not found.");
    }

    /*!create agent state lists */
    it->second->getPopulationData(population);
}

/**
 * @brief Loops through agents functions and register all variables
 * (variable has must also be tied to function name using the namespace thing in curve)
*/
bool CUDAAgentModel::step(const Simulation& simulation) {
    int nStreams = 1;
    std::string message_name;
    CurveNamespaceHash message_name_inp_hash = 0;
    CurveNamespaceHash message_name_outp_hash = 0;
    unsigned int messageList_Size = 0;

    // TODO: simulation.getMaxFunctionsPerLayer()
    for (unsigned int i = 0; i < simulation.getLayerCount(); i++) {
        int temp = static_cast<int>(simulation.getFunctionsAtLayer(i).size());
        nStreams = std::max(nStreams, temp);
    }

    /*!  Stream creations */
    cudaStream_t *stream = new cudaStream_t[nStreams];

    /*!  Stream initialisation */
    for (int j = 0; j < nStreams; j++)
        gpuErrchk(cudaStreamCreate(&stream[j]));


    /*! for each each sim layer, launch each agent function in its own stream */
    for (unsigned int i = 0; i < simulation.getLayerCount(); i++) {
        const auto& functions = simulation.getFunctionsAtLayer(i);

        int j = 0;
        // Sum the total number of threads being launched in the layer
        unsigned int totalThreads = 0;
        /*! for each func function - Loop through to do all mapping of agent and message variables */
        for (AgentFunctionDescription func_des : functions) {
            const CUDAAgent& cuda_agent = getCUDAAgent(func_des.getParent().getName());

            // check if a function has an input massage
            if (func_des.hasInputMessage()) {
                std::string inpMessage_name = func_des.getInputMessageName();
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name); printf("inp msg name: %s\n", inpMessage_name.c_str());
                cuda_message.mapRuntimeVariables(func_des);
            }

            // check if a function has an output massage
            if (func_des.hasOutputMessage()) {
                std::string outpMessage_name = func_des.getOutputMessageName();
                const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name); printf("inp msg name: %s\n", outpMessage_name.c_str());
                cuda_message.mapRuntimeVariables(func_des);
            }


            /**
             * Configure runtime access of the functions variables within the FLAME_API object
             */
            cuda_agent.mapRuntimeVariables(func_des);


            // Count total threads being launched
            totalThreads += cuda_agent.getMaximumListSize();
        }

        // Ensure RandomManager is the correct size to accomodate all threads to be launched
        RandomManager::resize(totalThreads);
        // Total threads is now used to provide kernel launches an offset to thread-safe thread-index
        totalThreads = 0;

        //! for each func function - Loop through to launch all agent functions
        for (AgentFunctionDescription func_des : functions) {
            std::string agent_name = func_des.getParent().getName();
            std::string func_name = func_des.getName();

            // check if a function has an output massage
            if (func_des.hasInputMessage()) {
                std::string inpMessage_name = func_des.getInputMessageName();
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                // message_name = inpMessage_name;

                // hash message name
                message_name_inp_hash = curveVariableRuntimeHash(inpMessage_name.c_str());

                messageList_Size = cuda_message.getMaximumListSize();
            }

            // check if a function has an output massage
            if (func_des.hasOutputMessage()) {
                std::string outpMessage_name = func_des.getOutputMessageName();
                // const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                // message_name = outpMessage_name;

                // hash message name
                message_name_outp_hash = curveVariableRuntimeHash(outpMessage_name.c_str());
            }

            const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

            // get the agent function
            FLAMEGPU_AGENT_FUNCTION_POINTER* agent_func = func_des.getFunction();

            // host_pointer
            FLAMEGPU_AGENT_FUNCTION_POINTER h_func_ptr;

            // cudaMemcpyFromSymbolAsync(&h_func_ptr, *agent_func, sizeof(FLAMEGPU_AGENT_FUNCTION_POINTER),0,cudaMemcpyDeviceToHost,stream[j]);
            cudaMemcpyFromSymbolAsync(&h_func_ptr, *agent_func, sizeof(FLAMEGPU_AGENT_FUNCTION_POINTER));

            int state_list_size = cuda_agent.getMaximumListSize();

            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            // calculate the grid block size for main agent function
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, agent_function_wrapper, 0, state_list_size);

            //! Round up according to CUDAAgent state list size
            gridSize = (state_list_size + blockSize - 1) / blockSize;

            // hash agent name
            CurveNamespaceHash agentname_hash = curveVariableRuntimeHash(agent_name.c_str());
            // hash function name
            CurveNamespaceHash funcname_hash = curveVariableRuntimeHash(func_name.c_str());

            // agent_function_wrapper << <gridSize, blockSize, 0, stream[j] >> > (agentname_hash + funcname_hash, h_func_ptr, state_list_size);
            agent_function_wrapper <<<gridSize, blockSize, 0, stream[j] >>>(agentname_hash + funcname_hash, message_name_inp_hash, message_name_outp_hash, h_func_ptr, state_list_size, messageList_Size, totalThreads);
            totalThreads += state_list_size;
            ++j;
        }
        // for each func function - Loop through to un-map all agent and message variables
        for (AgentFunctionDescription func_des : functions) {
            const CUDAAgent& cuda_agent = getCUDAAgent(func_des.getParent().getName());

            // check if a function has an output massage
            if (func_des.hasInputMessage()) {
                std::string inpMessage_name = func_des.getInputMessageName();
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                cuda_message.unmapRuntimeVariables(func_des);
            }

            // check if a function has an output massage
            if (func_des.hasOutputMessage()) {
                std::string outpMessage_name = func_des.getOutputMessageName();
                const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                cuda_message.unmapRuntimeVariables(func_des);
            }
            // const CUDAMessage& cuda_inpMessage = getCUDAMessage(func_des.getInputChild.getMessageName());
            // const CUDAMessage& cuda_outpMessage = getCUDAMessage(func_des.getOutputChild.getMessageName());

            // unmap the function variables
            cuda_agent.unmapRuntimeVariables(func_des);
        }

        // Execute all host functions attached to layer
        // TODO: Concurrency?
        for (auto &stepFn : simulation.getHostFunctionsAtLayer(i))
            stepFn(&this->host_api);

        // cudaDeviceSynchronize();
    }
    // stream deletion
    for (int j = 0; j < nStreams; ++j)
        gpuErrchk(cudaStreamDestroy(stream[j]));
    free(stream);


    // Execute step functions
    for (auto &stepFn : simulation.getStepFunctions())
        stepFn(&this->host_api);


    // Execute exit conditions
    for (auto &exitCdns : simulation.getExitConditions())
        if (exitCdns(&this->host_api) == EXIT)
            return false;
    return true;
}

/**
* @brief initialize CUDA params (e.g: set CUDA device)
* @warning not tested
*/
void CUDAAgentModel::init(void) {  // (int argc, char** argv) {
    cudaError_t cudaStatus;
    int device;
    int device_count;

    // default device
    device = 0;
    cudaStatus = cudaGetDeviceCount(&device_count);

    if (cudaStatus != cudaSuccess) {
        throw InvalidCUDAdevice("Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?");
    }
    if (device_count == 0) {
        throw InvalidCUDAdevice("Error no CUDA devices found!");
    }

    // Select device
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        throw InvalidCUDAdevice("Error setting CUDA device!");
    }
}

/**
* @brief simulates functions
* @param   object
* @return none
* @todo not yet completed
* @warning not tested
*/
void CUDAAgentModel::simulate(const Simulation& simulation) {
    if (agent_map.size() == 0)
        throw InvalidCudaAgentMapSize("CUDA agent map size is zero");  // recheck if this is really required

    // CUDAAgentMap::iterator it;

    // check any CUDAAgents with population size == 0
    // if they have executable functions then these can be ignored
    // if they have agent creations then buffer space must be allocated for them

    // Execute init functions
    for (auto &initFn : simulation.getInitFunctions())
        initFn(&this->host_api);

    for (unsigned int i = 0; simulation.getSimulationSteps() == 0 ? true : i < simulation.getSimulationSteps(); i++) {
        std::cout <<"step: " << i << std::endl;
        if (!step(simulation))
            break;
    }

    // Execute exit functions
    for (auto &exitFn : simulation.getExitFunctions())
        exitFn(&this->host_api);
}

const CUDAAgent& CUDAAgentModel::getCUDAAgent(std::string agent_name) const {
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        throw InvalidCudaAgent("CUDA agent not found.");
    }

    return *(it->second);
}

const CUDAMessage& CUDAAgentModel::getCUDAMessage(std::string message_name) const {
    CUDAMessageMap::const_iterator it;
    it = message_map.find(message_name);

    if (it == message_map.end()) {
        throw InvalidCudaMessage("CUDA message not found.");
    }

    return *(it->second);
}
