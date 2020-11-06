
/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */
#include "flamegpu/io/xmlWriter.h"
#include <sstream>
#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/gpu/CUDASimulation.h"

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

xmlWriter::xmlWriter(
    const std::string &model_name,
    const unsigned int &sim_instance_id, const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model,
    const unsigned int &iterations,
    const std::string &output_file,
    const Simulation *_sim_instance)
    : StateWriter(model_name, sim_instance_id, model, iterations, output_file, _sim_instance) {}

int xmlWriter::writeStates(bool prettyPrint) {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLNode * pRoot = doc.NewElement("states");
    doc.InsertFirstChild(pRoot);

    // Redundant for FLAMEGPU1 backwards compatibility
    tinyxml2::XMLElement * pElement = doc.NewElement("itno");
    pElement->SetText(iterations);
    pRoot->InsertEndChild(pElement);

    // Output config elements
    pElement = doc.NewElement("config");
    {
        // Sim config
        tinyxml2::XMLElement *pSimCfg = doc.NewElement("simulation");
        {
            const auto &sim_cfg = sim_instance->getSimulationConfig();
            tinyxml2::XMLElement *pListElement = nullptr;
            // Input file
            pListElement = doc.NewElement("input_file");
            pListElement->SetText(sim_cfg.input_file.c_str());
            pSimCfg->InsertEndChild(pListElement);
            // Steps
            pListElement = doc.NewElement("steps");
            pListElement->SetText(sim_cfg.steps);
            pSimCfg->InsertEndChild(pListElement);
            // Timing Output
            pListElement = doc.NewElement("timing");
            pListElement->SetText(sim_cfg.timing);
            pSimCfg->InsertEndChild(pListElement);
            // Random seed
            pListElement = doc.NewElement("random_seed");
            pListElement->SetText(sim_cfg.random_seed);
            pSimCfg->InsertEndChild(pListElement);
            // Verbose output
            pListElement = doc.NewElement("verbose");
            pListElement->SetText(sim_cfg.verbose);
            pSimCfg->InsertEndChild(pListElement);
            // Verbose output
            pListElement = doc.NewElement("console_mode");
            pListElement->SetText(sim_cfg.console_mode);
            pSimCfg->InsertEndChild(pListElement);
        }
        pElement->InsertEndChild(pSimCfg);

        // Cuda config
        if (auto *cudamodel_instance = dynamic_cast<const CUDASimulation*>(sim_instance)) {
            tinyxml2::XMLElement *pCUDACfg = doc.NewElement("cuda");
            {
                const auto &cuda_cfg = cudamodel_instance->getCUDAConfig();
                tinyxml2::XMLElement *pListElement = nullptr;
                // Input file
                pListElement = doc.NewElement("device_id");
                pListElement->SetText(cuda_cfg.device_id);
                pCUDACfg->InsertEndChild(pListElement);
            }
            pElement->InsertEndChild(pCUDACfg);
        }
    }
    pRoot->InsertEndChild(pElement);

    // Output stats elements
    pElement = doc.NewElement("stats");
    {
        tinyxml2::XMLElement *pListElement = nullptr;
        // Input file
        pListElement = doc.NewElement("step_count");
        pListElement->SetText(iterations);
        pElement->InsertEndChild(pListElement);
    }
    pRoot->InsertEndChild(pElement);

    pElement = doc.NewElement("environment");
    // for each environment property
    EnvironmentManager &env_manager = EnvironmentManager::getInstance();
    auto lock = env_manager.getSharedLock();
    const char *env_buffer = reinterpret_cast<const char *>(env_manager.getHostBuffer());
    for (auto &a : env_manager.getPropertiesMap()) {
        // If it is from this model
        if (a.first.first == sim_instance_id) {
            tinyxml2::XMLElement* pListElement = doc.NewElement(a.first.second.c_str());
            pListElement->SetAttribute("type", a.second.type.name());
                // Output properties
                std::stringstream ss;
                // Loop through elements, to construct csv string
                for (unsigned int el = 0; el < a.second.elements; ++el) {
                    if (a.second.type == std::type_index(typeid(float))) {
                        ss << *reinterpret_cast<const float*>(env_buffer + a.second.offset + (el * sizeof(float)));
                    } else if (a.second.type == std::type_index(typeid(double))) {
                        ss << *reinterpret_cast<const double*>(env_buffer + a.second.offset + (el * sizeof(double)));
                    } else if (a.second.type == std::type_index(typeid(int64_t))) {
                        ss << *reinterpret_cast<const int64_t*>(env_buffer + a.second.offset + (el * sizeof(int64_t)));
                    } else if (a.second.type == std::type_index(typeid(uint64_t))) {
                        ss << *reinterpret_cast<const uint64_t*>(env_buffer + a.second.offset + (el * sizeof(uint64_t)));
                    } else if (a.second.type == std::type_index(typeid(int32_t))) {
                        ss << *reinterpret_cast<const int32_t*>(env_buffer + a.second.offset + (el * sizeof(int32_t)));
                    } else if (a.second.type == std::type_index(typeid(uint32_t))) {
                        ss << *reinterpret_cast<const uint32_t*>(env_buffer + a.second.offset + (el * sizeof(uint32_t)));
                    } else if (a.second.type == std::type_index(typeid(int16_t))) {
                        ss << *reinterpret_cast<const int16_t*>(env_buffer + a.second.offset + (el * sizeof(int16_t)));
                    } else if (a.second.type == std::type_index(typeid(uint16_t))) {
                        ss << *reinterpret_cast<const uint16_t*>(env_buffer + a.second.offset + (el * sizeof(uint16_t)));
                    } else if (a.second.type == std::type_index(typeid(int8_t))) {
                        ss << static_cast<int32_t>(*reinterpret_cast<const int8_t*>(env_buffer + a.second.offset + (el * sizeof(int8_t))));  // Char outputs weird if being used as an integer
                    } else if (a.second.type == std::type_index(typeid(uint8_t))) {
                        ss << static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(env_buffer + a.second.offset + (el * sizeof(uint8_t))));  // Char outputs weird if being used as an integer
                    } else {
                        THROW TinyXMLError("Model contains environment property '%s' of unsupported type '%s', "
                            "in xmlWriter::writeStates()\n", a.first.second.c_str(), a.second.type.name());
                    }
                    if (el + 1 != a.second.elements)
                        ss << ",";
                }
            pListElement->SetText(ss.str().c_str());
            pElement->InsertEndChild(pListElement);
        }
    }
    pRoot->InsertEndChild(pElement);

    unsigned int populationSize;

    // for each agent types
    for (auto &agent : model_state) {
        // For each agent state
        for (auto &state : agent.second->getAgentDescription().states) {
            populationSize = agent.second->getStateMemory(state).getStateListSize();
            if (populationSize) {
                for (unsigned int i = 0; i < populationSize; ++i) {
                    // Create vars block
                    tinyxml2::XMLElement * pXagentElement = doc.NewElement("xagent");

                    AgentInstance instance = agent.second->getInstanceAt(i, state);
                    const auto &mm = agent.second->getAgentDescription().variables;

                    // Add agent's name to block
                    tinyxml2::XMLElement * pXagentNameElement = doc.NewElement("name");
                    pXagentNameElement->SetText(agent.first.c_str());
                    pXagentElement->InsertEndChild(pXagentNameElement);
                    // Add state's name to block
                    tinyxml2::XMLElement * pStateNameElement = doc.NewElement("state");
                    pStateNameElement->SetText(state.c_str());
                    pXagentElement->InsertEndChild(pStateNameElement);

                    // for each variable
                    for (auto iter_mm = mm.begin(); iter_mm != mm.end(); ++iter_mm) {
                        const std::string variable_name = iter_mm->first;

                        tinyxml2::XMLElement* pListElement = doc.NewElement(variable_name.c_str());
                        if (i == 0)
                            pListElement->SetAttribute("type", iter_mm->second.type.name());

                        // Output properties
                        std::stringstream ss;
                        // Loop through elements, to construct csv string
                        for (unsigned int el = 0; el < iter_mm->second.elements; ++el) {
                            if (iter_mm->second.type == std::type_index(typeid(float))) {
                                ss << instance.getVariable<float>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(double))) {
                                ss << instance.getVariable<double>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(int64_t))) {
                                ss << instance.getVariable<int64_t>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(uint64_t))) {
                                ss << instance.getVariable<uint64_t>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(int32_t))) {
                                ss << instance.getVariable<int32_t>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(uint32_t))) {
                                ss << instance.getVariable<uint32_t>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(int16_t))) {
                                ss << instance.getVariable<int16_t>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(uint16_t))) {
                                ss << instance.getVariable<uint16_t>(variable_name, el);
                            } else if (iter_mm->second.type == std::type_index(typeid(int8_t))) {
                                ss << static_cast<int32_t>(instance.getVariable<int8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                            } else if (iter_mm->second.type == std::type_index(typeid(uint8_t))) {
                                ss << static_cast<uint32_t>(instance.getVariable<uint8_t>(variable_name, el));  // Char outputs weird if being used as an integer
                            } else {
                                THROW TinyXMLError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                                    "in xmlWriter::writeStates()\n", agent.first.c_str(), variable_name.c_str(), iter_mm->second.type.name());
                            }
                            if (el + 1 != iter_mm->second.elements)
                                ss << ",";
                        }
                        pListElement->SetText(ss.str().c_str());
                        pXagentElement->InsertEndChild(pListElement);
                    }
                    // Insert xagent block into doc root
                    pRoot->InsertEndChild(pXagentElement);
                }
            }  // if state has agents
        }
    }

    tinyxml2::XMLError errorId = doc.SaveFile(outputFile.c_str(), !prettyPrint);
    XMLCheckResult(errorId);

    return tinyxml2::XML_SUCCESS;
}

