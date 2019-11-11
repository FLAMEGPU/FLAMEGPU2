/**
* @file Simulation.cpp
* @authors Paul
* @date
* @brief
*
* @see
* @warning
*/

#include <exception>
#include <algorithm>
#include <locale>

#include "flamegpu/sim/Simulation.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/io/statereader.h"
#include "flamegpu/io/statewriter.h"
#include "flamegpu/io/factory.h"
#include "flamegpu/runtime/utility/DeviceRandomArray.cuh"

Simulation::Simulation(const ModelDescription& model) : layers(), model_description(model) {
    simulation_steps = 1;
    DeviceRandomArray::init(static_cast<unsigned int>(DeviceRandomArray::seedFromTime()%UINT_MAX));
}


Simulation::~Simulation(void) {
    DeviceRandomArray::free();
}

const SimulationLayer::FunctionDescriptionVector& Simulation::getFunctionsAtLayer(const unsigned int &layer) const {
    if (layer >= layers.size()) {
        throw InvalidMemoryCapacity("Function layer doesn't exists!");  // out of bound index
    } else {
        return layers.at(layer).get().getAgentFunctions();
    }
}
const SimulationLayer::HostFunctionSet& Simulation::getHostFunctionsAtLayer(const unsigned int &layer) const {
    if (layer >= layers.size()) {
        throw InvalidMemoryCapacity("Function layer doesn't exists!");  // out of bound index
    } else {
        return layers.at(layer).get().getHostFunctions();
    }
}
const Simulation::InitFunctionSet& Simulation::getInitFunctions() const {
    return initFunctions;
}
const Simulation::StepFunctionSet& Simulation::getStepFunctions() const {
    return stepFunctions;
}
const Simulation::ExitFunctionSet& Simulation::getExitFunctions() const {
    return exitFunctions;
}
const Simulation::ExitConditionSet& Simulation::getExitConditions() const {
    return exitConditions;
}

unsigned int Simulation::addSimulationLayer(SimulationLayer &layer) {
    layers.push_back(layer);
    return static_cast<unsigned int>(layers.size())-1;
}

void Simulation::addInitFunction(const FLAMEGPU_INIT_FUNCTION_POINTER *func_p) {
    if (!initFunctions.insert(*func_p).second)
        throw InvalidHostFunc("Attempted to add same init function twice.");
}
void Simulation::addStepFunction(const FLAMEGPU_STEP_FUNCTION_POINTER *func_p) {
    if (!stepFunctions.insert(*func_p).second)
        throw InvalidHostFunc("Attempted to add same step function twice.");
}
void Simulation::addExitFunction(const FLAMEGPU_EXIT_FUNCTION_POINTER *func_p) {
    if (!exitFunctions.insert(*func_p).second)
        throw InvalidHostFunc("Attempted to add same exit function twice.");
}
void Simulation::addExitCondition(const FLAMEGPU_EXIT_CONDITION_POINTER *func_p) {
    if (!exitConditions.insert(*func_p).second)
        throw InvalidHostFunc("Attempted to add same exit condition twice.");
}

void Simulation::setSimulationSteps(unsigned int steps) {
    simulation_steps = steps;
}

unsigned int Simulation::getSimulationSteps() const {
    return simulation_steps;
}

unsigned int Simulation::getLayerCount() const {
    return (unsigned int) layers.size();
}

const ModelDescription& Simulation::getModelDescritpion() const {
    return model_description;
}

int Simulation::checkArgs(int argc, const char** argv, std::string &xml_model_path) {
    // These should really be in some kind of config struct
    // unsigned int device_id = 0;
    // unsigned int iterations = 0;

    // Required args
    if (argc < 2) {
        printHelp(argv[0]);
        return false;
    }
    xml_model_path = std::string(argv[1]);

    // Parse optional args
    int i = 2;
    for (; i < argc; i++) {
        // Get arg as lowercase
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), [](unsigned char c) { return std::use_facet< std::ctype<char>>(std::locale()).tolower(c); });
        // -in <string>, Specifies the input state file
        /*if (arg.compare("--in") == 0 || arg.compare("-i") == 0) {
            xml_model_path = std::string(argv[++i]);
            continue;
        }*/
        // -steps <uint>, The number of steps to be executed
        if (arg.compare("--steps") == 0 || arg.compare("-s") == 0) {
            // iterations = static_cast<int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -device <uint>, Uses the specified cuda device, defaults to 0
        if (arg.compare("--device") == 0 || arg.compare("-d") == 0) {
            // device_id = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0));
            continue;
        }
        // -random <uint>, Uses the specified random seed, defaults to clock
        if (arg.compare("--random") == 0 || arg.compare("-r") == 0) {
            // Reinitialise DeviceRandomArray state
            DeviceRandomArray::init(static_cast<unsigned int>(strtoul(argv[++i], nullptr, 0)));
            continue;
        }
        fprintf(stderr, "Unexpected argument: %s\n", arg.c_str());
        printHelp(argv[0]);
        return false;
    }
    return true;
}

void Simulation::printHelp(const char *executable) {
    printf("Usage: %s xml_input_file [-s steps] [-d device_id] [-r random_seed]\n", executable);
    printf("Optional Arguments:\n");
    const char *line_fmt = "%-18s %s\n";
    printf(line_fmt, "-s, --steps", "Number of simulation iterations");
    printf(line_fmt, "-d, --device", "GPU index");
    printf(line_fmt, "-r, --random", "DeviceRandomArray seed");
}

/**
* Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
* @param input    XML file path for agent initial configuration
*/
void Simulation::initialise(int argc, const char** argv) {
    // check input args
    std::string xml_model_path;
    if (!checkArgs(argc, argv, xml_model_path) && !xml_model_path.empty())
        exit(0);

    StateReader *read__ = ReaderFactory::createReader(model_description, xml_model_path.c_str());
    read__->parse();
}

/*
void Simulation::initialise(StateReader& read__) {
    read__.parse();
}
*/

/*
 * issues: only saves the last output, hardcoded, will be changed
 */
void Simulation::output(int /*argc*/, const char** /*argv*/) {
    // check input args
    // if (!checkArgs(argc, argv))
    // exit(0);
    const char* input =  "finalIteration.xml";  // argv[2];

    StateWriter *write__ = WriterFactory::createWriter(model_description, input);
    write__->writeStates();
}
