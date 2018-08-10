#ifndef STATEWRITER_H_
#define STATEWRITER_H_

/**
 * @file statereader.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */


#include "../xmlparser/tinyxml2.h"              //downloaded from https://github.com/leethomason/tinyxml2, the list of xml parsers : http://lars.ruoff.free.fr/xmlcpp/
#include "../exception/FGPUException.h"
#include "../pop/AgentPopulation.h"

using namespace std;
using namespace tinyxml2;

#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != XML_SUCCESS) { printf("Error: %i\n", a_eResult); return a_eResult; }
#endif

//TODO: Some example code of the handle class and an example function

class StateWriter;  // Forward declaration (class defined below)


class StateWriter
{

public:

	// -----------------------------------------------------------------------
	//  Constructors and Destructor
	// -----------------------------------------------------------------------

	StateWriter() {};

	~StateWriter() {};

	// -----------------------------------------------------------------------
	//  The interface
	// -----------------------------------------------------------------------

	/*!
	 *
	 */
	int writeStates(const ModelDescription &model_description, char* inputpath);


private:

	/* The copy constructor, you cannot call this directly */
	//StateWriter(const StateWriter&);

	/* The assignment operator, you cannot call this directly */
	//StateWriter& operator=(const StateWriter&);
};

int StateWriter::writeStates(const ModelDescription &model_description, char* outputpath)
{
	XMLDocument doc;

	XMLNode * pRoot = doc.NewElement("states");
	doc.InsertFirstChild(pRoot);

	XMLElement * pElement = doc.NewElement("itno");
	pElement->SetText(1); // get simulation step here - later
	pRoot->InsertEndChild(pElement);

	pElement = doc.NewElement("environment");
	pRoot->InsertEndChild(pElement);

	pElement = doc.NewElement("xagent");

	/* example
	for (int i = 0; i < populationSize; i++)
	{
		AgentInstance instance1 = model_description.getAgentPopulation(agentName).getInstanceAt(i, "default");
		if (instance1.getVariable<float>("x") != 0) { cout << i << " : " << instance1.getVariable<float>("x") << endl; }
	}

	for (auto &item: )
	{
		XMLElement* pListElement = doc->NewElement("name");
		pListElement->SetText(agentName);
	}
	*/
	pRoot->InsertEndChild(pElement);


	pRoot->InsertEndChild(pElement);
	XMLError errorId = doc.SaveFile(outputpath);
	XMLCheckResult(errorId);

	return XML_SUCCESS;
}



#endif /* STATEWRITER_H_ */
