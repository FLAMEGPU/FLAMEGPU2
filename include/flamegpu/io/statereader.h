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


#include <tinyxml2/tinyxml2.h>              //downloaded from https://github.com/leethomason/tinyxml2, the list of xml parsers : http://lars.ruoff.free.fr/xmlcpp/
#include <flamegpu/exception/FGPUException.h>
#include <flamegpu/pop/AgentPopulation.h>

using namespace std;
using namespace tinyxml2;

#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != XML_SUCCESS) { printf("XMLCheckResult Error: %i\n", a_eResult); return a_eResult; }
#endif

//TODO: Some example code of the handle class and an example function

class StateReader;  // Forward declaration (class defined below)

// Base class			
class StateReader {
public:

	// -----------------------------------------------------------------------
	//  Constructors and Destructor
	// -----------------------------------------------------------------------
	StateReader(const ModelDescription &model):model_description_(model) {};
	~StateReader() {};

	// -----------------------------------------------------------------------
	//  The interface
	// -----------------------------------------------------------------------

	virtual int parse( ) = 0;

	void setFileName(char* input) {	inputFile = input; }

	void setModelDesc(const ModelDescription &model_desc) {	model_description_ = model_desc; }

	StateReader& create(const ModelDescription &model, char *input);
	string getFileExt(const string& s);

protected:
	char* inputFile = "";
	ModelDescription model_description_;
};

// Derived classes
class xmlReader : public StateReader
{
public:
	xmlReader(const ModelDescription &model) : StateReader(model) {};
	int parse();
};

class binReader : public StateReader
{
public:
	binReader(const ModelDescription &model) : StateReader(model) {};
    int parse()
	{
		printf("to do, will exit now");
		exit(0);
	}
};

string StateReader::getFileExt(const string& s) {

	// Find the last position of '.' in given string
	size_t i = s.rfind('.', s.length());
	if (i != string::npos) {
		return(s.substr(i + 1, s.length() - i));
	}

	// In case of no extension return empty string
	return("");
}

StateReader& StateReader::create(const ModelDescription &model,char *input)
{
	string extension = getFileExt(input);
	StateReader *object_to_return = NULL;

	if (extension == "xml")
	{
		object_to_return = new xmlReader(model);
	}
	if (extension == "bin") 
	{
		object_to_return = new binReader(model);
	}

	return *object_to_return;
}

/**
* \brief parses the xml file
* \param source name of the inputfile
*/
int xmlReader::parse()
{
	XMLDocument doc;
	
	XMLError errorId = doc.LoadFile(inputFile);
	XMLCheckResult(errorId);

	printf("XML file '%s' loaded.\n", inputFile);

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

		const MemoryMap &m = model_description_.getAgentDescription(agentName).getMemoryMap();
		AgentInstance instance = model_description_.getAgentPopulation(agentName).getNextInstance("default");

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
