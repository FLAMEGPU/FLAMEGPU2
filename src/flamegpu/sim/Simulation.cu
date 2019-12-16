#include "flamegpu/sim/Simulation.h"

#include <algorithm>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/io/xmlWriter.h"
#include "flamegpu/io/factory.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "flamegpu/runtime/flamegpu_host_api.h"
#include "flamegpu/pop/AgentPopulation.h"


Simulation::Simulation(const ModelDescription& _model)
    : model(_model.model->clone())
    , host_api(std::make_unique<FLAMEGPU_HOST_API>(*this))
    , simulation_steps(0)
    , has_seed(false)
    , random_seed(0) {
}

void Simulation::initialise(int argc, const char** argv) {
    // check input args
    if (!checkArgs(argc, argv))
        exit(0);

    if (has_seed)
        RandomManager::getInstance().reseed(random_seed);

    // Build population vector
    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    for (auto &agent : model->agents) {
        auto a = std::make_shared<AgentPopulation>(*agent.second->description);
        pops.emplace(agent.first, a);
    }
    if (!xml_input_path.empty()) {
        StateReader *read__ = ReaderFactory::createReader(pops, xml_input_path.c_str());
        read__->parse();
        for (auto &agent : pops) {
            setPopulationData(*agent.second);
        }
    }

    // Call any derived class init methods
    _initialise();
}

void Simulation::setSimulationSteps(unsigned steps) {
    simulation_steps = steps;
}

unsigned Simulation::getSimulationSteps() const {
    return simulation_steps;
}

const ModelData& Simulation::getModelDescription() const {
    return *model;
}

/*
 * issues: only saves the last output, hardcoded, will be changed
 */
void Simulation::output(int /*argc*/, const char** /*argv*/) {
    // check input args
    // if (!checkArgs(argc, argv))
    // exit(0);
    const char* input = "finalIteration.xml";  // argv[2];

    // Build population vector
    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    for (auto &agent : model->agents) {
        auto a = std::make_shared<AgentPopulation>(*agent.second->description);
        getPopulationData(*a);
        pops.emplace(agent.first, a);
    }

    StateWriter *write__ = WriterFactory::createWriter(pops, input);  // TODO (pair model format with its data?)
    write__->writeStates();
}

int Simulation::checkArgs(int argc, const char** argv) {
    has_seed = false;
    // These should really be in some kind of config struct
    // unsigned int device_id = 0;
    // unsigned int iterations = 0;

    // Required args
    if (argc < 1) {
        printHelp(argv[0]);
        return false;
    }

    // Parse optional args
    int i = 1;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // -in <string>, Specifies the input state file
        if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
            xml_input_path = std::string(argv[++i]);
            continue;
        }
        // -steps <uint>, The number of steps to be executed
        if (arg.compare("--steps") == 0 || arg.compare("-s") == 0) {
            // iterations = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -random <uint>, Uses the specified random seed, defaults to clock
        if (arg.compare("--random") == 0 || arg.compare("-r") == 0) {
            // Reinitialise RandomManager state
            has_seed = true;
            random_seed = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // Test this arg with the derived class
        if (checkArgs_derived(argc-i, argv+i)) {
            continue;
        }
        fprintf(stderr, "Unexpected argument: %s\n", arg.c_str());
        printHelp(argv[0]);
        return false;
    }
    return true;
}

void Simulation::printHelp(const char* executable) {
    printf("Usage: %s [-s steps] [-d device_id] [-r random_seed]\n", executable);
    printf("Optional Arguments:\n");
    const char *line_fmt = "%-18s %s\n";
    printf(line_fmt, "-i, --in", "Initial state file (XML)");
    printf(line_fmt, "-s, --steps", "Number of simulation iterations");
    printf(line_fmt, "-r, --random", "RandomManager seed");
    printHelp_derived();
}
