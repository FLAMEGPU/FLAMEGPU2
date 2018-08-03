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
//#include "../model/AgentDescription.h"
 //#include "../pop/AgentPopulation.h"

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
	int readInitialStates(const AgentDescription &agent_description, char* inputpath);


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


int StateReader::readInitialStates(const AgentDescription &agent_description, char* inputpath)
{
	AgentPopulation population1(agent_description);

	//using Tinyxml Library:
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

	// Structure of the XML file:
	// -Element "states"        the root Element, which is the
	//                          FirstChildElement of the Document
	// --Element "itno"         child of the root states Element
	// --Element "environment"  child of the root states Element
	// --Element "xagent"       child of the root states Element
	// --- Text                 child of the xagent Element

	XMLNode* pRoot = doc.FirstChild();
	if (pRoot == nullptr)
		return XML_ERROR_FILE_READ_ERROR;

	XMLElement * pElement = pRoot->FirstChildElement("itno");
	if (pElement == nullptr)
		return XML_ERROR_PARSING_ELEMENT;

	int error;
	errorId = pElement->QueryIntText(&error);
	XMLCheckResult(errorId);

	pElement = pRoot->FirstChildElement("xagent");
	if (pElement == nullptr)
		return XML_ERROR_PARSING_ELEMENT;
	
	//88888888888888888888888
	// Note: either you need to know the name of the variables and their types (int or float) or you have to use the nextsiblingelement to retrieve these

/*	
	///// Solution 1 - not what we want
	for (XMLElement* child = pElement->FirstChildElement(); child; child = child->NextSiblingElement())
	{
		// do what you want with the element; in this case and as a simple example,
		// we try to convert its text to an int and display it on the standard output
		cout << child->GetText() << endl;
	}
*/


	XMLElement* pListElement = pElement->FirstChildElement("name");
	const char* agentName = pListElement->GetText();
	printf("Agent name: %s\n", agentName);

/*
	///// Solution 2 - good, but we have to know the variable's type/name
	//while (pListElement != nullptr)
	{
		int iOutInt;
		pListElement = pElement->FirstChildElement("x"); // "x"
		XMLCheckResult(errorId);
		pListElement->QueryIntText(&iOutInt);

		pListElement = pElement->FirstChildElement("y"); // "y"
		float fOutFloat;
		errorId = pListElement->QueryFloatText(&fOutFloat);
		XMLCheckResult(errorId);

		printf("%d %f\n", iOutInt, fOutFloat);
	}
*/

	///// Solution 3
	MemoryMap::const_iterator iter;
	const MemoryMap &m = agent_description.getMemoryMap();

	//while (pListElement != nullptr) {
		AgentInstance instance = population1.getNextInstance("default");

		for (iter = m.begin(); iter != m.end(); iter++)
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

	//88888888888888888888888
	/*
	std::vector<int> vecList;
	while (pListElement != nullptr)
	{
		AgentInstance instance = population1.getNextInstance("default");
		//instance.setVariable<float>("x",..);
		//instance.setVariable<float>("y",..);

		int iOutListValue;
		errorId = pListElement->QueryIntText(&iOutListValue);
		XMLCheckResult(errorId);

		float fOutFloat;
		errorId = pListElement->QueryFloatText(&fOutFloat);
		XMLCheckResult(errorId);

		vecList.push_back(iOutListValue);
		pListElement = pListElement->NextSiblingElement("Item");
	}

	/*
	XMLElement * pListElement = pElement->FirstChildElement("Item");
	std::vector<int> vecList;
	while (pListElement != nullptr)
	{
		int iOutListValue;
		eResult = pListElement->QueryIntText(&iOutListValue);
		XMLCheckResult(eResult);

		vecList.push_back(iOutListValue);
		pListElement = pListElement->NextSiblingElement("Item");
	}
	*/
	/*
	pElement = pRoot->FirstChildElement("FloatValue");
	if (pElement == nullptr) 
		return XML_ERROR_PARSING_ELEMENT;

	float fOutFloat;
	errorId = pElement->QueryFloatText(&fOutFloat);
	XMLCheckResult(errorId);
	*/

	return XML_SUCCESS;
}



#endif /* STATEREADER_H_ */
