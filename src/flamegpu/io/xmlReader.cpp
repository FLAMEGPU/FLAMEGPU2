
/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include "flamegpu/io/xmlReader.h"
#include <sstream>
#include <algorithm>
#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FGPUException.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/gpu/CUDAAgentModel.h"

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

xmlReader::xmlReader(
    const std::string &model_name,
    const unsigned int &sim_instance_id,
    const std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> &model_state,
    const std::string &input,
    Simulation *sim_instance)
    : StateReader(model_name, sim_instance_id, model_state, input, sim_instance) {}

/**
* \brief parses the xml file
*/
int xmlReader::parse() {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLError errorId = doc.LoadFile(inputFile.c_str());
    XMLCheckResult(errorId);

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

    // Read config data
    pElement = pRoot->FirstChildElement("config");
    if (pElement) {
        // Sim config
        if (sim_instance) {
            tinyxml2::XMLElement *pSimCfgBlock = pElement->FirstChildElement("simulation");
            for (auto simCfgElement = pSimCfgBlock->FirstChildElement(); simCfgElement; simCfgElement = simCfgElement->NextSiblingElement()) {
                std::string key = simCfgElement->Value();
                std::string val = simCfgElement->GetText();
                if (key == "input_file") {
                    if (inputFile != val)
                        printf("Warning: Input file '%s' refers to second input file '%s', this will not be loaded.\n", inputFile.c_str(), val.c_str());
                    // sim_instance->SimulationConfig().input_file = val;
                } else if (key == "steps") {
                    sim_instance->SimulationConfig().steps = static_cast<unsigned int>(stoull(val));
                } else if (key == "timing") {
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        sim_instance->SimulationConfig().timing = true;
                    } else if (val == "false") {
                        sim_instance->SimulationConfig().timing = false;
                    } else {
                        sim_instance->SimulationConfig().timing = static_cast<bool>(stoll(val));
                    }
                } else if (key == "random_seed") {
                    sim_instance->SimulationConfig().random_seed = static_cast<unsigned int>(stoull(val));
                } else if (key == "verbose") {
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        sim_instance->SimulationConfig().verbose = true;
                    } else if (val == "false") {
                        sim_instance->SimulationConfig().verbose = false;
                    } else {
                        sim_instance->SimulationConfig().verbose = static_cast<bool>(stoll(val));
                    }
                }  else {
                    fprintf(stderr, "Warning: Input file '%s' contains unexpected simulation config property '%s'.\n", inputFile.c_str(), key.c_str());
                }
            }
        }
        // CUDA config
        CUDAAgentModel *cudamodel_instance = dynamic_cast<CUDAAgentModel*>(sim_instance);
        if (cudamodel_instance) {
            tinyxml2::XMLElement *pCUDACfgBlock = pElement->FirstChildElement("cuda");
            for (auto cudaCfgElement = pCUDACfgBlock->FirstChildElement(); cudaCfgElement; cudaCfgElement = cudaCfgElement->NextSiblingElement()) {
                std::string key = cudaCfgElement->Value();
                std::string val = cudaCfgElement->GetText();
                if (key == "device_id") {
                    cudamodel_instance->CUDAConfig().device_id = static_cast<unsigned int>(stoull(val));
                }  else {
                    fprintf(stderr, "Warning: Input file '%s' contains unexpected cuda config property '%s'.\n", inputFile.c_str(), key.c_str());
                }
            }
        }
    } else {
        // No warning, environment node is not mandatory
    }

    // Read environment data
    EnvironmentManager &env_manager = EnvironmentManager::getInstance();
    pElement = pRoot->FirstChildElement("environment");
    if (pElement) {
        for (auto envElement = pElement->FirstChildElement(); envElement; envElement = envElement->NextSiblingElement()) {
            const char *key = envElement->Value();
            std::stringstream ss(envElement->GetText());
            std::string token;
            const EnvironmentManager::NamePair np = { sim_instance_id , std::string(key) };
            if (env_manager.contains(np)) {
                const std::type_index val_type = env_manager.type(np);
                const auto elements = env_manager.length(np);
                unsigned int el = 0;
                while (getline(ss, token, ',')) {
                    if (val_type == std::type_index(typeid(float))) {
                        env_manager.set<float>(np, el++, stof(token));
                    } else if (val_type == std::type_index(typeid(double))) {
                        env_manager.set<double>(np, el++, stod(token));
                    } else if (val_type == std::type_index(typeid(int64_t))) {
                        env_manager.set<int64_t>(np, el++, stoll(token));
                    } else if (val_type == std::type_index(typeid(uint64_t))) {
                        env_manager.set<uint64_t>(np, el++, stoull(token));
                    } else if (val_type == std::type_index(typeid(int32_t))) {
                        env_manager.set<int32_t>(np, el++, static_cast<int32_t>(stoll(token)));
                    } else if (val_type == std::type_index(typeid(uint32_t))) {
                        env_manager.set<uint32_t>(np, el++, static_cast<uint32_t>(stoull(token)));
                    } else if (val_type == std::type_index(typeid(int16_t))) {
                        env_manager.set<int16_t>(np, el++, static_cast<int16_t>(stoll(token)));
                    } else if (val_type == std::type_index(typeid(uint16_t))) {
                        env_manager.set<uint16_t>(np, el++, static_cast<uint16_t>(stoull(token)));
                    } else if (val_type == std::type_index(typeid(int8_t))) {
                        env_manager.set<int8_t>(np, el++, static_cast<int8_t>(stoll(token)));
                    } else if (val_type == std::type_index(typeid(uint8_t))) {
                        env_manager.set<uint8_t>(np, el++, static_cast<uint8_t>(stoull(token)));
                    } else {
                        THROW TinyXMLError("Model contains environment property '%s' of unsupported type '%s', "
                            "in xmlReader::parse()\n", key, val_type.name());
                    }
                }
                if (el != elements) {
                    fprintf(stderr, "Warning: Environment array property '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                        key, elements, inputFile.c_str(), el);
                }
            } else {
                fprintf(stderr, "Warning: Input file '%s' contains unexpected environment property '%s'.\n", inputFile.c_str(), key);
            }
        }
    } else {
        fprintf(stderr, "Warning: Input file '%s' does not contain environment node.\n", inputFile.c_str());
    }

    // Count how many of each agent are in the file and resize state lists
    std::unordered_map<std::string, unsigned int> cts;
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        cts[std::string(pElement->FirstChildElement("name")->GetText())]++;
    }
    // Resize state lists
    for (auto &agt : cts) {
        if (agt.second > AgentPopulation::DEFAULT_POPULATION_SIZE)
            model_state.at(agt.first)->setStateListCapacity(agt.second);
    }

    // Read in agent data
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        // Find agent name
        tinyxml2::XMLElement* pNameElement = pElement->FirstChildElement("name");
        const char* agentName = pNameElement->GetText();
        // Find agent state, use initial state if not set (means its old fgpu1 input file)
        tinyxml2::XMLElement* pStateElement = pElement->FirstChildElement("state");
        const char* agentState = pStateElement ? pStateElement->GetText() : model_state.at(agentName)->getAgentDescription().initial_state.c_str();
        const auto &agentVariables = model_state.at(agentName)->getAgentDescription().variables;
        // Create instace to store variable data in
        AgentInstance instance = model_state.at(agentName)->getNextInstance(agentState);
        bool hasWarnedElements = false;
        bool hasWarnedMissingVar = false;
        // Iterate agent variables
        for (auto iter = agentVariables.begin(); iter != agentVariables.end(); ++iter) {
            const std::string variable_name = iter->first;
            const auto &var_data = iter->second;
            tinyxml2::XMLElement* pVarElement = pElement->FirstChildElement(variable_name.c_str());
            if (pVarElement) {
                // Put string for the variable into a string stream
                std::stringstream ss(pVarElement->GetText());
                std::string token;
                // Iterate elements of the stringstream
                unsigned int el = 0;
                while (getline(ss, token, ',')) {
                    if (var_data.type == std::type_index(typeid(float))) {
                        instance.setVariable<float>(variable_name, el++, stof(token));
                    } else if (var_data.type == std::type_index(typeid(double))) {
                        instance.setVariable<double>(variable_name, el++, stod(token));
                    } else if (var_data.type == std::type_index(typeid(int64_t))) {
                        instance.setVariable<int64_t>(variable_name, el++, stoll(token));
                    } else if (var_data.type == std::type_index(typeid(uint64_t))) {
                        instance.setVariable<uint64_t>(variable_name, el++, stoull(token));
                    } else if (var_data.type == std::type_index(typeid(int32_t))) {
                        instance.setVariable<int32_t>(variable_name, el++, static_cast<int32_t>(stoll(token)));
                    } else if (var_data.type == std::type_index(typeid(uint32_t))) {
                        instance.setVariable<uint32_t>(variable_name, el++, static_cast<uint32_t>(stoull(token)));
                    } else if (var_data.type == std::type_index(typeid(int16_t))) {
                        instance.setVariable<int16_t>(variable_name, el++, static_cast<int16_t>(stoll(token)));
                    } else if (var_data.type == std::type_index(typeid(uint16_t))) {
                        instance.setVariable<uint16_t>(variable_name, el++, static_cast<uint16_t>(stoull(token)));
                    } else if (var_data.type == std::type_index(typeid(int8_t))) {
                        instance.setVariable<int8_t>(variable_name, el++, static_cast<int8_t>(stoll(token)));
                    } else if (var_data.type == std::type_index(typeid(uint8_t))) {
                        instance.setVariable<uint8_t>(variable_name, el++, static_cast<uint8_t>(stoull(token)));
                    } else {
                        THROW TinyXMLError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                            "in xmlReader::parse()\n", agentName, variable_name.c_str(), var_data.type.name());
                    }
                }
                // Warn if var is wrong length
                if (el != var_data.elements && !hasWarnedElements) {
                    fprintf(stderr, "Warning: Agent '%s' variable '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                        agentName, variable_name.c_str(), var_data.elements, inputFile.c_str(), el);
                    hasWarnedElements = true;
                }
            } else if (!hasWarnedMissingVar) {
                fprintf(stderr, "Warning: Agent '%s' variable '%s' is missing from, input file '%s'.\n",
                    agentName, variable_name.c_str(), inputFile.c_str());
                hasWarnedMissingVar = true;
            }
        }
    }

    return tinyxml2::XML_SUCCESS;
}
