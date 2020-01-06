
/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include <tinyxml2/tinyxml2.h>              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include <flamegpu/exception/FGPUException.h>
#include <flamegpu/model/AgentDescription.h>
#include <flamegpu/pop/AgentPopulation.h>
#include "flamegpu/io/xmlWriter.h"

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

xmlWriter::xmlWriter(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model, const unsigned int &iterations, const char* output_file) 
    : StateWriter(model, iterations, output_file) {}

int xmlWriter::writeStates() {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLNode * pRoot = doc.NewElement("states");
    doc.InsertFirstChild(pRoot);

    tinyxml2::XMLElement * pElement = doc.NewElement("itno");
    pElement->SetText(iterations);  // get simulation step here - later
    pRoot->InsertEndChild(pElement);

    pElement = doc.NewElement("environment");
    pRoot->InsertEndChild(pElement);

    int populationSize;

    // for each agent types
    for (auto &agent : model_state) {
        const char* agentName = agent.first.c_str();

        populationSize = agent.second->getStateMemory().getStateListSize();

        // for each agent
        for (int i = 0; i < populationSize; i++) {
            pElement = doc.NewElement("xagent");

            tinyxml2::XMLElement* pListElement = doc.NewElement("name");
            pListElement->SetText(agentName);
            pElement->InsertEndChild(pListElement);

            AgentInstance instance = agent.second->getInstanceAt(i, "default");
            const auto &mm = agent.second->getAgentDescription().variables;

            // for each variable
            for (auto iter_mm = mm.begin(); iter_mm != mm.end(); ++iter_mm) {
                const std::string variable_name = iter_mm->first;

                pListElement = doc.NewElement(variable_name.c_str());

                if (iter_mm->second.type == std::type_index(typeid(float)))
                    pListElement->SetText(instance.getVariable<float>(variable_name));
                else if (iter_mm->second.type == std::type_index(typeid(double)))
                    pListElement->SetText(instance.getVariable<double>(variable_name));
                else if (iter_mm->second.type == std::type_index(typeid(int)))
                    pListElement->SetText(instance.getVariable<int>(variable_name));
                else if (iter_mm->second.type == std::type_index(typeid(bool)))
                    pListElement->SetText(instance.getVariable<bool>(variable_name));

                pElement->InsertEndChild(pListElement);
            }

            pRoot->InsertEndChild(pElement);
        }
    }

    tinyxml2::XMLError errorId = doc.SaveFile(outputFile.c_str());
    XMLCheckResult(errorId);

    return tinyxml2::XML_SUCCESS;
}

