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
	
	for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent"))
	{
		if (pElement == nullptr)
			return XML_ERROR_PARSING_ELEMENT;

		XMLElement* pListElement = pElement->FirstChildElement("name");
		const char* agentName = pListElement->GetText();

		const MemoryMap &m = model_description.getAgentDescription(agentName).getMemoryMap();
		AgentInstance instance = model_description.getAgentPopulation(agentName).getNextInstance("default");

		for (MemoryMap::const_iterator iter = m.begin(); iter != m.end(); iter++)
		{
			float outFloat;
			double outDouble;
			int outInt;
			bool outBool;

			const std::string variable_name = iter->first;
			pListElement = pElement->FirstChildElement(variable_name.c_str());
			XMLCheckResult(errorId);

			if (iter->second == typeid(float)) {
				errorId = pListElement->QueryFloatText(&outFloat);

				instance.setVariable<float>(variable_name, outFloat);
			}
			else if (iter->second == typeid(double)) {
				errorId = pListElement->QueryDoubleText(&outDouble);
				XMLCheckResult(errorId);

				instance.setVariable<double>(variable_name, outDouble);
			}
			else if (iter->second == typeid(int)) {
				errorId = pListElement->QueryIntText(&outInt);
				XMLCheckResult(errorId);

				instance.setVariable<int>(variable_name, outInt);
			}
			else if (iter->second == typeid(bool)) {
				errorId = pListElement->QueryBoolText(&outBool);
				XMLCheckResult(errorId);

				instance.setVariable<bool>(variable_name, outBool);
			}
		}
	}
	
	return XML_SUCCESS;
}



#endif /* STATEREADER_H_ */
