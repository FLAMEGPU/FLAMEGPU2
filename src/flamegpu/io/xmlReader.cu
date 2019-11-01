
/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <string>
#include <tinyxml2/tinyxml2.h>              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/io/xmlReader.h"

#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { printf("XMLCheckResult Error: %i\n", a_eResult); return a_eResult; }
#endif

xmlReader::xmlReader(const ModelDescription &model, const char* input) : StateReader(model, input) {}

/**
* \brief parses the xml file
*/
int xmlReader::parse() {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLError errorId = doc.LoadFile(inputFile.c_str());
    XMLCheckResult(errorId);

    printf("XML file '%s' loaded.\n", inputFile.c_str());

    tinyxml2::XMLNode* pRoot = doc.FirstChild();
    if (pRoot == nullptr)
        return tinyxml2::XML_ERROR_FILE_READ_ERROR;

    tinyxml2::XMLElement * pElement = pRoot->FirstChildElement("itno");
    if (pElement == nullptr)
        return tinyxml2::XML_ERROR_PARSING_ELEMENT;

    int error;
    errorId = pElement->QueryIntText(&error);
    XMLCheckResult(errorId);

    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        if (pElement == nullptr)
            return tinyxml2::XML_ERROR_PARSING_ELEMENT;

        tinyxml2::XMLElement* pListElement = pElement->FirstChildElement("name");
        const char* agentName = pListElement->GetText();

        const MemoryMap &m = model_description_.getAgentDescription(agentName).getMemoryMap();
        AgentInstance instance = model_description_.getAgentPopulation(agentName).getNextInstance("default");

        for (MemoryMap::const_iterator iter = m.begin(); iter != m.end(); iter++) {
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
            } else if (iter->second == typeid(double)) {
                errorId = pListElement->QueryDoubleText(&outDouble);
                XMLCheckResult(errorId);

                instance.setVariable<double>(variable_name, outDouble);
            } else if (iter->second == typeid(int)) {
                errorId = pListElement->QueryIntText(&outInt);
                XMLCheckResult(errorId);

                instance.setVariable<int>(variable_name, outInt);
            } else if (iter->second == typeid(bool)) {
                errorId = pListElement->QueryBoolText(&outBool);
                XMLCheckResult(errorId);

                instance.setVariable<bool>(variable_name, outBool);
            }
        }
    }

    return tinyxml2::XML_SUCCESS;
}
