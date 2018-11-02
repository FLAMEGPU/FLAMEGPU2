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


Simulation::Simulation(const ModelDescription& model) : model_description(model), layers()
{
    simulation_steps = 1;
}


Simulation::~Simulation(void)
{
}


const FunctionDescriptionVector& Simulation::getFunctionsAtLayer(unsigned int layer) const
{
    if (layer>=layers.size())
        throw InvalidMemoryCapacity("Function layer doesn't exists!"); // out of bound index
    else
    {
        return layers.at(layer).get().getAgentFunctions(); 
    }
}


unsigned int Simulation::addSimulationLayer(SimulationLayer& layer)
{
    layers.push_back(layer);
    return static_cast<unsigned int>(layers.size())-1;
}

void Simulation::setSimulationSteps(unsigned int steps)
{
    simulation_steps = steps;
}

unsigned int Simulation::getSimulationSteps() const
{
	return simulation_steps;
}

unsigned int Simulation::getLayerCount() const
{
    return (unsigned int) layers.size();
}

const ModelDescription& Simulation::getModelDescritpion() const
{
    return model_description;
}

string Simulation::getFileExt(const string& s) {

	// Find the last position of '.' in given string
	size_t i = s.rfind('.', s.length());
	if (i != string::npos) {
		return(s.substr(i + 1, s.length() - i));
	}
	// In case of no extension return empty string
	return("");
}


int Simulation::checkArgs(int argc, char** argv) {
	//Check args
	printf("FLAMEGPU Console mode\n");
	if (argc < 2)
	{
		printf("Usage: main [XML model data] [Iterations] [Optional CUDA device ID]\n");
		return false;
	}

	return true;
}

/**
* Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
* @param input	XML file path for agent initial configuration
*/
void Simulation::initialise(int argc, char** argv)
{
	//check input args
	if (!checkArgs(argc, argv))
		exit(0);
	const char* input = argv[1];

	//Factory *factory = new ReaderFactory;
	//std::unique_ptr<Factory> ReaderFactory;

	ReaderFactory1 *rf = new ReaderFactory1;

	string extension = getFileExt(input);
	StateReader *read__ = NULL;

	if (extension == "xml")
		read__ = rf->create_xml(model_description, input);
	else 
		printf("Format not supported");

	//std::unique_ptr<StateReader> read_;
	//StateReader &read__(read_->create(model_description, input));
	//read__.setFileName(input);
	//read__.setModelDesc(model_description);

	read__->parse();

	// todo : move factory class to outside (later)
	//We could use an if condition here to find out which derived class to create
	/*
	xmlReader read_ (model);

	read_.setFileName(input);
	read_.setModelDesc(model);
	read_.parse();
	*/
}

/*
void Simulation::initialise(StateReader& read__)
{
	read__.parse();
}
*/


/*
 * issues: only saves the last output, hardcoded, will be changed
 */
void Simulation::output(int argc, char** argv)
{
	//check input args
	if (!checkArgs(argc, argv))
		exit(0);
	const char* input = "finalIteration.xml";

	WriterFactory1 *wf = new WriterFactory1;

	string extension = getFileExt(input);
	StateWriter *write__ = NULL;

	if (extension == "xml")
		write__ = wf->write_xml(model_description, input);
	else
		printf("Format not supported");

	write__->writeStates();

	/*
	//read initial states
	StateWriter statewrite_;
	statewrite_.writeStates(model_description, output);
	*/
}
