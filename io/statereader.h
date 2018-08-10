#ifndef STATEREADER_H_
#define STATEREADER_H_

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

class StateReader;  // Forward declaration (class defined below)


class StateReader
{

public:

	// -----------------------------------------------------------------------
	//  Constructors and Destructor
	// -----------------------------------------------------------------------

	StateReader() {};

	~StateReader() {};

	// -----------------------------------------------------------------------
	//  The interface
	// -----------------------------------------------------------------------

	/*!
	 *
	 */
	 //virtual void parse(const char &source) = 0;
	int readInitialStates(const ModelDescription &model_description, char* inputpath);


private:

	/* The copy constructor, you cannot call this directly */
	//StateReader(const StateReader&);

	/* The assignment operator, you cannot call this directly */
	//StateReader& operator=(const StateReader&);
};

/**
* \brief
* \param source name of the inputfile
*/
//void StateReader::parse(const inputSource &source)

int StateReader::readInitialStates(const ModelDescription &model_description, char* inputpath)
{
	XMLDocument doc;

	XMLError errorId = doc.LoadFile(inputpath);
	XMLCheckResult(errorId);

	//int errorID = doc.ErrorID(); // not required

	//printf("XML file '%s' loaded. ErrorID=%d\n", inputpath, errorID);
	printf("XML file '%s' loaded.\n", inputpath);

	/* Pointer to file */
	FILE* fp = fopen(inputpath, "r");

	/* Open config file to read-only */
	if (!fp)
	{
		printf("Error opening initial states\n");
		exit(0);
	}

	/* Close the file */
	fclose(fp);

	XMLNode* pRoot = doc.FirstChild();
	if (pRoot == nullptr)
		return XML_ERROR_FILE_READ_ERROR;

	XMLElement * pElement = pRoot->FirstChildElement("itno");
	if (pElement == nullptr)
		return XML_ERROR_PARSING_ELEMENT;

	int error;
	errorId = pElement->QueryIntText(&error);
	XMLCheckResult(errorId);
	
	
	// then check the agent name agent the tag name to avoid doing the inialization agent.

	// note: either pass the vector of varnames to the statereader , create the objects on the main and pass the varnames
	// or look at the population , maybe instead of passing the flamemodel, we can pass the population. But we need to be able to get the instances
	// or add these to the flame model
	

	for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent"))
	{
		if (pElement == nullptr)
			return XML_ERROR_PARSING_ELEMENT;

		XMLElement* pListElement = pElement->FirstChildElement("name");
		const char* agentName = pListElement->GetText();

		printf("Agent name: %s\n", agentName);	

		const MemoryMap &m = model_description.getAgentDescription(agentName).getMemoryMap();
		AgentInstance instance = model_description.getAgentPopulation(agentName).getNextInstance("default");

		for (MemoryMap::const_iterator iter = m.begin(); iter != m.end(); iter++)
		{
			float outFloat;
			int outInt;

			const std::string variable_name = iter->first;
			pListElement = pElement->FirstChildElement(variable_name.c_str());
			XMLCheckResult(errorId);

			if (iter->second == typeid(float)) {
				errorId = pListElement->QueryFloatText(&outFloat);
				XMLCheckResult(errorId);

				instance.setVariable<float>(variable_name, outFloat);
				printf(": %f\n", outFloat);
			}
			else {
				pListElement->QueryIntText(&outInt);
				XMLCheckResult(errorId);

				instance.setVariable<int>(variable_name, outInt);
				printf(": %d\n", outInt);
			}
			printf(": %s\n", variable_name);
		}
	}
	
	return XML_SUCCESS;
}



#endif /* STATEREADER_H_ */
