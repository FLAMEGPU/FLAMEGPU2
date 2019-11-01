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

#include <flamegpu/sim/Simulation.h>
#include <flamegpu/model/ModelDescription.h>
#include <flamegpu/io/statereader.h>
#include <flamegpu/io/statewriter.h>
#include "flamegpu/io/factory.h"

Simulation::Simulation(const ModelDescription& model) : layers(), model_description(model) {
    simulation_steps = 1;
}

Simulation::~Simulation(void) {
}

const FunctionDescriptionVector& Simulation::getFunctionsAtLayer(unsigned int layer) const {
    if (layer>=layers.size())
        throw InvalidMemoryCapacity("Function layer doesn't exists!"); // out of bound index
    else {
        return layers.at(layer).get().getAgentFunctions();
    }
}

unsigned int Simulation::addSimulationLayer(SimulationLayer& layer) {
    layers.push_back(layer);
    return static_cast<unsigned int>(layers.size())-1;
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

int Simulation::checkArgs(int argc, char** argv) {
    // Check args
    printf("FLAMEGPU Console mode\n");
    if (argc < 2) {
        printf("Usage: %s [XML model data] [Iterations] [Optional CUDA device ID]\n", argv[0]);
        return false;
    }

    return true;
}

/**
* Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
* @param input    XML file path for agent initial configuration
*/
void Simulation::initialise(int argc, char** argv) {
    // check input args
    if (!checkArgs(argc, argv))
        exit(0);
    const char* input = argv[1];

    StateReader *read__ = ReaderFactory::createReader(model_description, input);
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
void Simulation::output(int argc, char** argv) {
    // check input args
    if (!checkArgs(argc, argv))
        exit(0);
    const char* input =  "finalIteration.xml";// argv[2];

    StateWriter *write__ = WriterFactory::createWriter(model_description, input);
    write__->writeStates();
}
