#include "flamegpu/sim/Simulation.h"

#include <algorithm>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/io/xmlWriter.h"
#include "flamegpu/io/factory.h"
#include "flamegpu/runtime/utility/RandomManager.cuh"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/util/nvtx.h"


Simulation::Simulation(const ModelDescription& _model)
    : model(_model.model->clone()) { }

void Simulation::initialise(int argc, const char** argv) {
    NVTX_RANGE("Simulation::initialise");
    config = Config();  // Reset to defaults
    resetDerivedConfig();
    // check input args
    if (argc)
        if (!checkArgs(argc, argv))
            exit(EXIT_FAILURE);
    applyConfig();
}

void Simulation::applyConfig() {
    // Call derived class config stuff first
    applyConfig_derived();
    // The cuda part of this should really be a seperate class triggered by derived cfg
    RandomManager::getInstance().reseed(config.random_seed);
    // Build population vector
    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    for (auto &agent : model->agents) {
        auto a = std::make_shared<AgentPopulation>(*agent.second->description);
        pops.emplace(agent.first, a);
    }
    if (!config.xml_input_file.empty()) {
        StateReader *read__ = ReaderFactory::createReader(model->name, pops, config.xml_input_file.c_str());
        if (read__) {
            read__->parse();
            for (auto &agent : pops) {
                setPopulationData(*agent.second);
            }
        }
    }
}

const ModelData& Simulation::getModelDescription() const {
    return *model;
}

/*
 * issues: only saves the last output, hardcoded, will be changed
 */
void Simulation::exportData(const std::string &path) {
    // Build population vector
    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    for (auto &agent : model->agents) {
        auto a = std::make_shared<AgentPopulation>(*agent.second->description);
        getPopulationData(*a);
        pops.emplace(agent.first, a);
    }

    StateWriter *write__ = WriterFactory::createWriter(model->name, pops, getStepCounter(), path);
    write__->writeStates();
}

int Simulation::checkArgs(int argc, const char** argv) {
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
            config.xml_input_file = std::string(argv[++i]);
            continue;
        }
        // -steps <uint>, The number of steps to be executed
        if (arg.compare("--steps") == 0 || arg.compare("-s") == 0) {
            config.steps = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -random <uint>, Uses the specified random seed, defaults to clock
        if (arg.compare("--random") == 0 || arg.compare("-r") == 0) {
            // Reinitialise RandomManager state
            config.random_seed = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -v/--verbose, Verbose FLAME GPU output.
        if (arg.compare("--verbose") == 0 || arg.compare("-v") == 0) {
            config.verbose = true;
            continue;
        }
        // -t/--timing, Output timing information to stdout
        if (arg.compare("--timing") == 0 || arg.compare("-t") == 0) {
            config.timing = true;
            continue;
        }
        // Test this arg with the derived class
        if (checkArgs_derived(argc, argv, i)) {
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
    printf(line_fmt, "-i, --in <file.xml>", "Initial state file (XML)");
    printf(line_fmt, "-s, --steps <steps>", "Number of simulation iterations");
    printf(line_fmt, "-r, --random <seed>", "RandomManager seed");
    printf(line_fmt, "-v, --verbose", "Verbose FLAME GPU output");
    printf(line_fmt, "-t, --timing", "Output timing information to stdout");
    printHelp_derived();
}

Simulation::Config &Simulation::SimulationConfig() {
    return config;
}
const Simulation::Config &Simulation::getSimulationConfig() const {
    return config;
}
