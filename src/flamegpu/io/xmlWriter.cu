
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
#include <flamegpu/pop/AgentPopulation.h>
#include "flamegpu/io/xmlWriter.h"

using namespace std;
using namespace tinyxml2;

#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != XML_SUCCESS) { printf("XMLCheckResult Error: %i\n", a_eResult); return a_eResult; }
#endif

xmlWriter::xmlWriter(const ModelDescription &model, const char* output) : StateWriter(model, output) {};

int xmlWriter::writeStates()
{
    XMLDocument doc;

    XMLNode * pRoot = doc.NewElement("states");
    doc.InsertFirstChild(pRoot);

    XMLElement * pElement = doc.NewElement("itno");
    pElement->SetText(1); // get simulation step here - later
    pRoot->InsertEndChild(pElement);

    pElement = doc.NewElement("environment");
    pRoot->InsertEndChild(pElement);


    int populationSize;

    const AgentMap &am = model_description_.getAgentMap();

    // for each agent types
    for (AgentMap::const_iterator iter_am = am.begin(); iter_am != am.end(); iter_am++)
    {
        const char* agentName = iter_am->first.c_str();

        populationSize = model_description_.getAgentPopulation(agentName).getStateMemory().getStateListSize();

        // for each agent
        for (int i = 0; i < populationSize; i++)
        {

            pElement = doc.NewElement("xagent");

            XMLElement* pListElement = doc.NewElement("name");
            pListElement->SetText(agentName);
            pElement->InsertEndChild(pListElement);

            AgentInstance instance = model_description_.getAgentPopulation(agentName).getInstanceAt(i, "default");
            const MemoryMap &mm = model_description_.getAgentDescription(agentName).getMemoryMap();

            // for each variable
            for (MemoryMap::const_iterator iter_mm = mm.begin(); iter_mm != mm.end(); iter_mm++)
            {
                const std::string variable_name = iter_mm->first;

                pListElement = doc.NewElement(variable_name.c_str());

                if (iter_mm->second == typeid(float))
                    pListElement->SetText(instance.getVariable<float>(variable_name));
                else if (iter_mm->second == typeid(double))
                    pListElement->SetText(instance.getVariable<double>(variable_name));
                else if (iter_mm->second == typeid(int))
                    pListElement->SetText(instance.getVariable<int>(variable_name));
                else if (iter_mm->second == typeid(bool))
                    pListElement->SetText(instance.getVariable<bool>(variable_name));

                pElement->InsertEndChild(pListElement);
            }

            pRoot->InsertEndChild(pElement);
        }
    }

    XMLError errorId = doc.SaveFile(outputFile.c_str());
    XMLCheckResult(errorId);

    return XML_SUCCESS;
}

