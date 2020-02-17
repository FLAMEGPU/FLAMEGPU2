#include "flamegpu/gpu/CUDAAgentModel.h"

#include <algorithm>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

CUDAAgentModel::CUDAAgentModel(const ModelDescription& _model)
    : Simulation(_model)
    , step_count(0)
    , agent_map()
    , curve(Curve::getInstance())
    , message_map()
    , rng(RandomManager::getInstance())
    , scatter(CUDAScatter::getInstance(0)) {
    rng.increaseSimCounter();
    scatter.increaseSimCounter();

    // populate the CUDA agent map
    const auto &am = model->agents;
    // create new cuda agent and add to the map
    for (auto it = am.cbegin(); it != am.cend(); ++it) {
        agent_map.emplace(it->first, std::make_unique<CUDAAgent>(*it->second));
    }  // insert into map using value_type

    // populate the CUDA message map
    const auto &mm = model->messages;
    // create new cuda message and add to the map
    for (auto it_m = mm.cbegin(); it_m != mm.cend(); ++it_m) {
        message_map.emplace(it_m->first, std::make_unique<CUDAMessage>(*it_m->second));
    }
    // Populate the environment properties in constant Cache
    {
        EnvironmentManager::getInstance().init(model->name, *model->environment);
    }
}

CUDAAgentModel::~CUDAAgentModel() {
    rng.decreaseSimCounter();
    scatter.decreaseSimCounter();
    // unique pointers cleanup by automatically
    // Drop all constants from the constant cache linked to this model
    EnvironmentManager::getInstance().free(model->name);
}

bool CUDAAgentModel::step() {
    step_count++;
    int nStreams = 1;
    std::string message_name;
    Curve::NamespaceHash message_name_inp_hash = 0;
    Curve::NamespaceHash message_name_outp_hash = 0;
    // hash model name
    const Curve::NamespaceHash modelname_hash = curve.variableRuntimeHash(model->name.c_str());

    const void *d_messagelist_metadata = nullptr;

    // TODO: simulation.getMaxFunctionsPerLayer()
    for (auto lyr = model->layers.begin(); lyr != model->layers.end(); ++lyr) {
        int temp = static_cast<int>((*lyr)->agent_functions.size());
        nStreams = std::max(nStreams, temp);
    }

    /*!  Stream creations */
    cudaStream_t *stream = new cudaStream_t[nStreams];

    /*!  Stream initialisation */
    for (int j = 0; j < nStreams; j++)
        gpuErrchk(cudaStreamCreate(&stream[j]));

    // Reset message list flags
    for (auto m =  message_map.begin(); m != message_map.end(); ++m) {
        m->second->setTruncateMessageListFlag();
    }

    /*! for each each sim layer, launch each agent function in its own stream */
    for (auto lyr = model->layers.begin(); lyr != model->layers.end(); ++lyr) {
        const auto& functions = (*lyr)->agent_functions;

        int j = 0;  // Track stream id
        // Sum the total number of threads being launched in the layer
        unsigned int totalThreads = 0;
        /*! for each func function - Loop through to do all mapping of agent and message variables */
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock()->description;
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            const CUDAAgent& cuda_agent = getCUDAAgent(func_agent->getName());
            flamegpu_internal::CUDAScanCompaction::resizeAgents(cuda_agent.getStateSize(func_des->initial_state), j);

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                // Construct PBM here if required!!
                cuda_message.buildIndex();
                // Map variables after, as index building can swap arrays
                cuda_message.mapReadRuntimeVariables(*func_des);
            }

            // check if a function has an output massage
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                // Resize message list if required
                cuda_message.resize(cuda_agent.getStateSize(func_des->initial_state), j);
                cuda_message.mapWriteRuntimeVariables(*func_des);
                flamegpu_internal::CUDAScanCompaction::resizeMessages(cuda_agent.getStateSize(func_des->initial_state), j);
                // Zero the scan flag that will be written to
                if (func_des->message_output_optional)
                    flamegpu_internal::CUDAScanCompaction::zeroMessages(j);  // Always default stream currently
            }


            /**
             * Configure runtime access of the functions variables within the FLAME_API object
             */
            cuda_agent.mapRuntimeVariables(*func_des);

            // Zero the scan flag that will be written to
            if (func_des->has_agent_death)
                flamegpu_internal::CUDAScanCompaction::zeroAgents(0);  // Always default stream currently

            // Count total threads being launched
            totalThreads += cuda_agent.getMaximumListSize();
            ++j;
        }

        // Ensure RandomManager is the correct size to accomodate all threads to be launched
        rng.resize(totalThreads);
        // Total threads is now used to provide kernel launches an offset to thread-safe thread-index
        totalThreads = 0;
        j = 0;
        //! for each func function - Loop through to launch all agent functions
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock();
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            std::string agent_name = func_agent->name;
            std::string func_name = func_des->name;

            // check if a function has an output massage
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                // message_name = inpMessage_name;

                // hash message name
                message_name_inp_hash = curve.variableRuntimeHash(inpMessage_name.c_str());

                d_messagelist_metadata = cuda_message.getMetaDataDevicePtr();
            }

            // check if a function has an output massage
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                // const CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                // message_name = outpMessage_name;

                // hash message name
                message_name_outp_hash =  curve.variableRuntimeHash(outpMessage_name.c_str());
            }

            const CUDAAgent& cuda_agent = getCUDAAgent(agent_name);

            int state_list_size = cuda_agent.getStateSize(func_des->initial_state);

            int blockSize = 0;  // The launch configurator returned block size
            int minGridSize = 0;  // The minimum grid size needed to achieve the // maximum occupancy for a full device // launch
            int gridSize = 0;  // The actual grid size needed, based on input size

            // calculate the grid block size for main agent function
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func_des->func, 0, state_list_size);

            //! Round up according to CUDAAgent state list size
            gridSize = (state_list_size + blockSize - 1) / blockSize;

            // hash agent name
            Curve::NamespaceHash agentname_hash = curve.variableRuntimeHash(agent_name.c_str());
            // hash function name
            Curve::NamespaceHash funcname_hash = curve.variableRuntimeHash(func_name.c_str());

            (func_des->func)<<<gridSize, blockSize, 0, stream[j] >>>(modelname_hash, agentname_hash + funcname_hash, message_name_inp_hash, message_name_outp_hash, state_list_size, d_messagelist_metadata, totalThreads, j);
            gpuErrchkLaunch();

            totalThreads += state_list_size;
            ++j;
        }

        j = 0;
        // for each func function - Loop through to un-map all agent and message variables
        for (const std::shared_ptr<AgentFunctionData> &func_des : functions) {
            auto func_agent = func_des->parent.lock()->description;
            if (!func_agent) {
                THROW InvalidAgentFunc("Agent function refers to expired agent.");
            }
            CUDAAgent& cuda_agent = getCUDAAgent(func_agent->getName());

            // check if a function has an input message
            if (auto im = func_des->message_input.lock()) {
                std::string inpMessage_name = im->name;
                const CUDAMessage& cuda_message = getCUDAMessage(inpMessage_name);
                cuda_message.unmapRuntimeVariables(*func_des);
            }

            // check if a function has an output message
            if (auto om = func_des->message_output.lock()) {
                std::string outpMessage_name = om->name;
                CUDAMessage& cuda_message = getCUDAMessage(outpMessage_name);
                cuda_message.unmapRuntimeVariables(*func_des);
                cuda_message.swap(func_des->message_output_optional, j);
                cuda_message.clearTruncateMessageListFlag();
                cuda_message.setPBMConstructionRequiredFlag();
            }
            // const CUDAMessage& cuda_inpMessage = getCUDAMessage(func_des.getInputChild.getMessageName());
            // const CUDAMessage& cuda_outpMessage = getCUDAMessage(func_des.getOutputChild.getMessageName());

            // unmap the function variables
            cuda_agent.unmapRuntimeVariables(*func_des);

            // Process agent death (has agent death check is handled by the method)
            cuda_agent.process_death(*func_des, j);

            ++j;
        }

        // Execute all host functions attached to layer
        // TODO: Concurrency?
        for (auto &stepFn : (*lyr)->host_functions)
            stepFn(this->host_api.get());
        // cudaDeviceSynchronize();
    }
    // stream deletion
    for (int j = 0; j < nStreams; ++j)
        gpuErrchk(cudaStreamDestroy(stream[j]));
    delete[] stream;


    // Execute step functions
    for (auto &stepFn : model->stepFunctions)
        stepFn(this->host_api.get());


    // Execute exit conditions
    for (auto &exitCdns : model->exitConditions)
        if (exitCdns(this->host_api.get()) == EXIT)
            return false;
    return true;
}

void CUDAAgentModel::simulate() {
    if (agent_map.size() == 0) {
        THROW InvalidCudaAgentMapSize("Simulation has no agents, in CUDAAgentModel::simulate().");  // recheck if this is really required
    }
    // CUDAAgentMap::iterator it;

    // check any CUDAAgents with population size == 0
    // if they have executable functions then these can be ignored
    // if they have agent creations then buffer space must be allocated for them

    // Execute init functions
    for (auto &initFn : model->initFunctions)
        initFn(this->host_api.get());

    for (unsigned int i = 0; getSimulationConfig().steps == 0 ? true : i < getSimulationConfig().steps; i++) {
        // std::cout <<"step: " << i << std::endl;
        if (!step())
            break;
    }

    // Execute exit functions
    for (auto &exitFn : model->exitFunctions)
        exitFn(this->host_api.get());
}

void CUDAAgentModel::setPopulationData(AgentPopulation& population) {
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDAAgentModel::setPopulationData()",
            population.getAgentName().c_str());
    }

    /*! create agent state lists */
    it->second->setPopulationData(population);
}

void CUDAAgentModel::getPopulationData(AgentPopulation& population) {
    CUDAAgentMap::iterator it;
    it = agent_map.find(population.getAgentName());

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("Error: Agent ('%s') was not found, "
            "in CUDAAgentModel::getPopulationData()",
            population.getAgentName().c_str());
    }

    /*!create agent state lists */
    it->second->getPopulationData(population);
}

CUDAAgent& CUDAAgentModel::getCUDAAgent(const std::string& agent_name) const {
    CUDAAgentMap::const_iterator it;
    it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDAAgentModel::getCUDAAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

AgentInterface& CUDAAgentModel::getAgent(const std::string& agent_name) {
    auto it = agent_map.find(agent_name);

    if (it == agent_map.end()) {
        THROW InvalidCudaAgent("CUDA agent ('%s') not found, in CUDAAgentModel::getAgent().",
            agent_name.c_str());
    }

    return *(it->second);
}

CUDAMessage& CUDAAgentModel::getCUDAMessage(const std::string& message_name) const {
    CUDAMessageMap::const_iterator it;
    it = message_map.find(message_name);

    if (it == message_map.end()) {
        THROW InvalidCudaMessage("CUDA message ('%s') not found, in CUDAAgentModel::getCUDAMessage().",
            message_name.c_str());
    }

    return *(it->second);
}

bool CUDAAgentModel::checkArgs_derived(int argc, const char** argv, int &i) {
    // Get arg as lowercase
    std::string arg(argv[i]);
    std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
    // -device <uint>, Uses the specified cuda device, defaults to 0
    if ((arg.compare("--device") == 0 || arg.compare("-d") == 0) && argc > i+1) {
        config.device_id = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
        return true;
    }
    return false;
}

void CUDAAgentModel::printHelp_derived() {
    const char *line_fmt = "%-18s %s\n";
    printf("CUDA Model Optional Arguments:\n");
    printf(line_fmt, "-d, --device", "GPU index");
}

void CUDAAgentModel::applyConfig_derived() {
    cudaError_t cudaStatus;
    int device_count;

    // default device
    cudaStatus = cudaGetDeviceCount(&device_count);

    if (cudaStatus != cudaSuccess) {
        THROW InvalidCUDAdevice("Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?");
    }
    if (device_count == 0) {
        THROW InvalidCUDAdevice("Error no CUDA devices found!");
    }

    // Select device
    if (config.device_id >= device_count) {
        THROW InvalidCUDAdevice("Error setting CUDA device to '%d', only %d available!", config.device_id, device_count);
    }
    cudaStatus = cudaSetDevice(static_cast<int>(config.device_id));
    if (cudaStatus != cudaSuccess) {
        THROW InvalidCUDAdevice("Unknown error setting CUDA device to '%d'. (%d available)", config.device_id, device_count);
    }
}

void CUDAAgentModel::resetDerivedConfig() {
    this->config = CUDAAgentModel::Config();
    resetStepCounter();
}


CUDAAgentModel::Config &CUDAAgentModel::CUDAConfig() {
    return config;
}
const CUDAAgentModel::Config &CUDAAgentModel::getCUDAConfig() const {
    return config;
}

unsigned int CUDAAgentModel::getStepCounter() {
    return step_count;
}
void CUDAAgentModel::resetStepCounter() {
    step_count = 0;
}
