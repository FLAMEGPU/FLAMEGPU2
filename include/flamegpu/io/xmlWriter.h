#ifndef XMLWRITER_H_
#define XMLWRITER_H_

/**
 * @file xmlwriter.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <string>
#include  <flamegpu/io/statewriter.h>
#include "flamegpu/model/ModelDescription.h"

// Derived classes
class xmlWriter : public StateWriter
{
public:
	xmlWriter(const ModelDescription &model, const char* output);
	int writeStates();
};


#endif /* XMLWRITER_H_ */
