
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
#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { printf("XMLCheckResult Error: %i\n", a_eResult); return a_eResult; }
#endif

xmlWriter::xmlWriter(const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model, const char* output) : StateWriter(model, output) {}

int xmlWriter::writeStates() {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLNode * pRoot = doc.NewElement("states");
    doc.InsertFirstChild(pRoot);

    tinyxml2::XMLElement * pElement = doc.NewElement("itno");
    pElement->SetText(1);  // get simulation step here - later
    pRoot->InsertEndChild(pElement);

    pElement = doc.NewElement("environment");
    pRoot->InsertEndChild(pElement);

    int populationSize;
    
    // for each agent types
    for (auto &agent:model_state) {
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

