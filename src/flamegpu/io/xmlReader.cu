
/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <string>
#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/io/xmlReader.h"

#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { FGPUException::setLocation(__FILE__, __LINE__);\
    switch (a_eResult) { \
    case tinyxml2::XML_ERROR_FILE_NOT_FOUND : \
    case tinyxml2::XML_ERROR_FILE_COULD_NOT_BE_OPENED : \
        throw InvalidInputFile("TinyXML error: File could not be opened.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_FILE_READ_ERROR : \
        throw InvalidInputFile("TinyXML error: File could not be read.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_PARSING_ELEMENT : \
    case tinyxml2::XML_ERROR_PARSING_ATTRIBUTE : \
    case tinyxml2::XML_ERROR_PARSING_TEXT : \
    case tinyxml2::XML_ERROR_PARSING_CDATA : \
    case tinyxml2::XML_ERROR_PARSING_COMMENT : \
    case tinyxml2::XML_ERROR_PARSING_DECLARATION : \
    case tinyxml2::XML_ERROR_PARSING_UNKNOWN : \
    case tinyxml2::XML_ERROR_PARSING : \
        throw TinyXMLError("TinyXML error: Error parsing file.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_EMPTY_DOCUMENT : \
        throw TinyXMLError("TinyXML error: XML_ERROR_EMPTY_DOCUMENT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_MISMATCHED_ELEMENT : \
        throw TinyXMLError("TinyXML error: XML_ERROR_MISMATCHED_ELEMENT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_CAN_NOT_CONVERT_TEXT : \
        throw TinyXMLError("TinyXML error: XML_CAN_NOT_CONVERT_TEXT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_NO_TEXT_NODE : \
        throw TinyXMLError("TinyXML error: XML_NO_TEXT_NODE\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ELEMENT_DEPTH_EXCEEDED : \
        throw TinyXMLError("TinyXML error: XML_ELEMENT_DEPTH_EXCEEDED\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_COUNT : \
        throw TinyXMLError("TinyXML error: XML_ERROR_COUNT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_NO_ATTRIBUTE: \
        throw TinyXMLError("TinyXML error: XML_NO_ATTRIBUTE\n Error code: %d", a_eResult); \
    case tinyxml2::XML_WRONG_ATTRIBUTE_TYPE : \
        throw TinyXMLError("TinyXML error: XML_WRONG_ATTRIBUTE_TYPE\n Error code: %d", a_eResult); \
    default: \
        throw TinyXMLError("TinyXML error: Unrecognised error code\n Error code: %d", a_eResult); \
    } \
}
#endif

xmlReader::xmlReader(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state, const char* input) : StateReader(model_state, input) {}

/**
* \brief parses the xml file
*/
int xmlReader::parse() {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLError errorId = doc.LoadFile(inputFile.c_str());
    XMLCheckResult(errorId);

    printf("XML file '%s' loaded.\n", inputFile.c_str());

    tinyxml2::XMLNode* pRoot = doc.FirstChild();
    if (pRoot == nullptr) {
        THROW TinyXMLError("TinyXML error: Error parsing doc %s.", inputFile.c_str());
    }

    tinyxml2::XMLElement * pElement = pRoot->FirstChildElement("itno");
    if (pElement == nullptr) {
        THROW TinyXMLError("TinyXML error: Error parsing element %s.", inputFile.c_str());
    }

    int error;
    errorId = pElement->QueryIntText(&error);
    XMLCheckResult(errorId);

    // Count how many of each agent are in the file
    std::unordered_map<std::string, unsigned int> cts;
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        cts[std::string(pElement->FirstChildElement("name")->GetText())]++;
    }
    // Resize state lists
    for (auto &agt : cts) {
        model_state.at(agt.first)->setStateListCapacity(agt.second);
    }


    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        if (pElement == nullptr) {
            THROW TinyXMLError("TinyXML error: Error parsing element %s.", inputFile.c_str());
        }

        tinyxml2::XMLElement* pListElement = pElement->FirstChildElement("name");
        const char* agentName = pListElement->GetText();

        const auto &m = model_state.at(agentName)->getAgentDescription().variables;
        AgentInstance instance = model_state.at(agentName)->getNextInstance(ModelData::DEFAULT_STATE);

        for (auto iter = m.begin(); iter != m.end(); ++iter) {
            float outFloat;
            double outDouble;
            int outInt;
            bool outBool;

            const std::string variable_name = iter->first;
            pListElement = pElement->FirstChildElement(variable_name.c_str());
            XMLCheckResult(errorId);

            if (iter->second.type == std::type_index(typeid(float))) {
                errorId = pListElement->QueryFloatText(&outFloat);

                instance.setVariable<float>(variable_name, outFloat);
            } else if (iter->second.type == std::type_index(typeid(double))) {
                errorId = pListElement->QueryDoubleText(&outDouble);
                XMLCheckResult(errorId);

                instance.setVariable<double>(variable_name, outDouble);
            } else if (iter->second.type == std::type_index(typeid(int))) {
                errorId = pListElement->QueryIntText(&outInt);
                XMLCheckResult(errorId);

                instance.setVariable<int>(variable_name, outInt);
            } else if (iter->second.type == std::type_index(typeid(bool))) {
                errorId = pListElement->QueryBoolText(&outBool);
                XMLCheckResult(errorId);

                instance.setVariable<bool>(variable_name, outBool);
            }
        }
    }

    return tinyxml2::XML_SUCCESS;
}
