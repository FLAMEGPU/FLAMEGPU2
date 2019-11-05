#ifndef STATEWRITER_H_
#define STATEWRITER_H_

/**
 * @file statewriter.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */


#include <tinyxml2/tinyxml2.h>              //downloaded from https://github.com/leethomason/tinyxml2, the list of xml parsers : http://lars.ruoff.free.fr/xmlcpp/
#include <flamegpu/exception/FGPUException.h>
#include "flamegpu/model/ModelDescription.h"


class StateWriter
{

public:

	// -----------------------------------------------------------------------
	//  Constructors and Destructor
	// -----------------------------------------------------------------------

	StateWriter(const ModelDescription &model, const char* output) : model_description_(model), outputFile(std::string(output)) {};
	~StateWriter() {};

	// -----------------------------------------------------------------------
	//  The interface
	// -----------------------------------------------------------------------

	virtual int writeStates() = 0;

protected:
	ModelDescription model_description_;
	std::string outputFile;
};

#endif /* STATEWRITER_H_ */
