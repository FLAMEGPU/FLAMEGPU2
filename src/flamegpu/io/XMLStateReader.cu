#include "flamegpu/io/XMLStateReader.h"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cstdio>
#include <vector>
#include <functional>
#include <memory>
#include <string>

#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/simulation/AgentVector.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/simulation/CUDASimulation.h"

namespace flamegpu {
namespace io {

#ifndef XMLCheckResult
/**
 * Macro function for converting a tinyxml2 return code to an exception
 * @param a_eResult The tinyxml2 return code
 */
#define XMLCheckResult(a_eResult) if (a_eResult != tinyxml2::XML_SUCCESS) { exception::FLAMEGPUException::setLocation(__FILE__, __LINE__);\
    switch (a_eResult) { \
    case tinyxml2::XML_ERROR_FILE_NOT_FOUND : \
    case tinyxml2::XML_ERROR_FILE_COULD_NOT_BE_OPENED : \
        THROW exception::InvalidInputFile("TinyXML error: File could not be opened.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_FILE_READ_ERROR : \
        THROW exception::InvalidInputFile("TinyXML error: File could not be read.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_PARSING_ELEMENT : \
    case tinyxml2::XML_ERROR_PARSING_ATTRIBUTE : \
    case tinyxml2::XML_ERROR_PARSING_TEXT : \
    case tinyxml2::XML_ERROR_PARSING_CDATA : \
    case tinyxml2::XML_ERROR_PARSING_COMMENT : \
    case tinyxml2::XML_ERROR_PARSING_DECLARATION : \
    case tinyxml2::XML_ERROR_PARSING_UNKNOWN : \
    case tinyxml2::XML_ERROR_PARSING : \
        THROW exception::TinyXMLError("TinyXML error: Error parsing file.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_EMPTY_DOCUMENT : \
        THROW exception::TinyXMLError("TinyXML error: XML_ERROR_EMPTY_DOCUMENT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_MISMATCHED_ELEMENT : \
        THROW exception::TinyXMLError("TinyXML error: XML_ERROR_MISMATCHED_ELEMENT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_CAN_NOT_CONVERT_TEXT : \
        THROW exception::TinyXMLError("TinyXML error: XML_CAN_NOT_CONVERT_TEXT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_NO_TEXT_NODE : \
        THROW exception::TinyXMLError("TinyXML error: XML_NO_TEXT_NODE\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ELEMENT_DEPTH_EXCEEDED : \
        THROW exception::TinyXMLError("TinyXML error: XML_ELEMENT_DEPTH_EXCEEDED\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_COUNT : \
        THROW exception::TinyXMLError("TinyXML error: XML_ERROR_COUNT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_NO_ATTRIBUTE: \
        THROW exception::TinyXMLError("TinyXML error: XML_NO_ATTRIBUTE\n Error code: %d", a_eResult); \
    case tinyxml2::XML_WRONG_ATTRIBUTE_TYPE : \
        THROW exception::TinyXMLError("TinyXML error: XML_WRONG_ATTRIBUTE_TYPE\n Error code: %d", a_eResult); \
    default: \
        THROW exception::TinyXMLError("TinyXML error: Unrecognised error code\n Error code: %d", a_eResult); \
    } \
}
#endif

void XMLStateReader::parse(const std::string &inputFile, const std::shared_ptr<const ModelData> &model, Verbosity verbosity) {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLError errorId = doc.LoadFile(inputFile.c_str());
    XMLCheckResult(errorId);

    tinyxml2::XMLNode* pRoot = doc.FirstChild();
    if (pRoot == nullptr) {
        THROW exception::TinyXMLError("TinyXML error: Error parsing doc %s.", inputFile.c_str());
    }

    // Read config data
    tinyxml2::XMLElement* pElement = pRoot->FirstChildElement("config");
    if (pElement) {
        // Sim config
        tinyxml2::XMLElement *pSimCfgBlock = pElement->FirstChildElement("simulation");
        if (pSimCfgBlock) {
            for (auto simCfgElement = pSimCfgBlock->FirstChildElement(); simCfgElement; simCfgElement = simCfgElement->NextSiblingElement()) {
                std::string key = simCfgElement->Value();
                std::string val = simCfgElement->GetText() ? simCfgElement->GetText() : "";
                if (key == "input_file") {
                    if (inputFile != val && !val.empty())
                        if (verbosity > Verbosity::Quiet)
                            fprintf(stderr, "Warning: Input file '%s' refers to second input file '%s', this will not be loaded.\n", inputFile.c_str(), val.c_str());
                    // sim_instance->SimulationConfig().input_file = val;
                } else if (key == "step_log_file" ||
                           key == "exit_log_file" ||
                           key == "common_log_file") {
                    simulation_config.emplace(key, val);
                } else if (key == "truncate_log_files" ||
#ifdef FLAMEGPU_VISUALISATION
                           key == "console_mode" ||
#endif
                           key == "timing") {
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        simulation_config.emplace(key, true);
                    } else if (val == "false") {
                        simulation_config.emplace(key, false);
                    } else {
                        simulation_config.emplace(key, static_cast<bool>(stoll(val)));
                    }
                } else if (key == "random_seed") {
                    simulation_config.emplace(key, static_cast<uint64_t>(stoull(val)));
                } else if (key == "steps") {
                    simulation_config.emplace(key, static_cast<unsigned int>(stoull(val)));
                } else if (key == "verbosity") {
                    simulation_config.emplace(key, static_cast<flamegpu::Verbosity>(stoull(val)));
                }  else if (verbosity > Verbosity::Quiet) {
                    fprintf(stderr, "Warning: Input file '%s' contains unexpected simulation config property '%s'.\n", inputFile.c_str(), key.c_str());
                }
#ifndef FLAMEGPU_VISUALISATION
                if (key == "console_mode") {
                    if (verbosity > Verbosity::Quiet)
                        fprintf(stderr, "Warning: Cannot configure 'console_mode' with input file '%s', FLAMEGPU2 library has not been built with visualisation support enabled.\n", inputFile.c_str());
                }
#endif
            }
        }
        // CUDA config
        tinyxml2::XMLElement *pCUDACfgBlock = pElement->FirstChildElement("cuda");
        if (pCUDACfgBlock) {
            for (auto cudaCfgElement = pCUDACfgBlock->FirstChildElement(); cudaCfgElement; cudaCfgElement = cudaCfgElement->NextSiblingElement()) {
                std::string key = cudaCfgElement->Value();
                std::string val = cudaCfgElement->GetText();
                if (key == "device_id") {
                    cuda_config.emplace(key, static_cast<int>(stoull(val)));
                } else if (key == "inLayerConcurrency") {
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        cuda_config.emplace(key, true);
                    } else if (val == "false") {
                        cuda_config.emplace(key, false);
                    } else {
                        cuda_config.emplace(key, static_cast<bool>(stoll(val)));
                    }
                } else if (verbosity > Verbosity::Quiet) {
                    fprintf(stderr, "Warning: Input file '%s' contains unexpected cuda config property '%s'.\n", inputFile.c_str(), key.c_str());
                }
            }
        }
    } else {
        // No warning, environment node is not mandatory
    }

    // Read environment data
    pElement = pRoot->FirstChildElement("environment");
    if (pElement) {
        auto& env_props = model->environment->properties;
        for (auto envElement = pElement->FirstChildElement(); envElement; envElement = envElement->NextSiblingElement()) {
            const char *key = envElement->Value();
            std::stringstream ss(envElement->GetText());
            std::string token;
            const auto it = env_props.find(std::string(key));
            if (it == env_props.end()) {
                THROW exception::TinyXMLError("Input file contains unrecognised environment property '%s',"
                    "in XMLStateReader::parse()\n", key);
            }
            const std::type_index val_type = it->second.data.type;
            const auto elements = it->second.data.elements;
            unsigned int el = 0;
            while (getline(ss, token, ',')) {
                if (el == 0) {
                    if (!env_init.emplace(std::string(key), detail::Any(it->second.data)).second) {
                        THROW exception::TinyXMLError("Input file contains environment property '%s' multiple times, "
                            "in XMLStateReader::parse()\n", key);
                    }
                } else if (el >= it->second.data.elements) {
                    THROW exception::RapidJSONError("Input file contains environment property '%s' too many elements, expected %u,"
                        "in XMLStateReader::parse()\n", key, it->second.data.elements);
                }
                const auto ei_it = env_init.find(key);
                if (val_type == std::type_index(typeid(float))) {
                    static_cast<float*>(const_cast<void*>(ei_it->second.ptr))[el++] = stof(token);
                } else if (val_type == std::type_index(typeid(double))) {
                    static_cast<double*>(const_cast<void*>(ei_it->second.ptr))[el++] = stod(token);
                } else if (val_type == std::type_index(typeid(int64_t))) {
                    static_cast<int64_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = stoll(token);
                } else if (val_type == std::type_index(typeid(uint64_t))) {
                    static_cast<uint64_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = stoull(token);
                } else if (val_type == std::type_index(typeid(int32_t))) {
                    static_cast<int32_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = static_cast<int32_t>(stoll(token));
                } else if (val_type == std::type_index(typeid(uint32_t))) {
                    static_cast<uint32_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = static_cast<uint32_t>(stoull(token));
                } else if (val_type == std::type_index(typeid(int16_t))) {
                    static_cast<int16_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = static_cast<int16_t>(stoll(token));
                } else if (val_type == std::type_index(typeid(uint16_t))) {
                    static_cast<uint16_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = static_cast<uint16_t>(stoull(token));
                } else if (val_type == std::type_index(typeid(int8_t))) {
                    static_cast<int8_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = static_cast<int8_t>(stoll(token));
                } else if (val_type == std::type_index(typeid(uint8_t))) {
                    static_cast<uint8_t*>(const_cast<void*>(ei_it->second.ptr))[el++] = static_cast<uint8_t>(stoull(token));
                } else {
                    THROW exception::TinyXMLError("Model contains environment property '%s' of unsupported type '%s', "
                        "in XMLStateReader::parse()\n", key, val_type.name());
                }
            }
            if (el != elements && verbosity > Verbosity::Quiet) {
                fprintf(stderr, "Warning: Environment array property '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                    key, elements, inputFile.c_str(), el);
            }
        }
    } else if (verbosity > Verbosity::Quiet) {
        fprintf(stderr, "Warning: Input file '%s' does not contain environment node.\n", inputFile.c_str());
    }

    // Read macro environment data
    pElement = pRoot->FirstChildElement("macro_environment");
    if (pElement) {
        auto& env_macro_props = model->environment->macro_properties;
        for (auto envElement = pElement->FirstChildElement(); envElement; envElement = envElement->NextSiblingElement()) {
            const char *key = envElement->Value();
            std::stringstream ss(envElement->GetText());
            std::string token;
            const auto it = env_macro_props.find(std::string(key));
            if (it == env_macro_props.end()) {
                THROW exception::TinyXMLError("Input file contains unrecognised macro environment property '%s',"
                    "in XMLStateReader::parse()\n", key);
            }
            const std::type_index val_type = it->second.type;
            const unsigned int elements = std::accumulate(it->second.elements.begin(), it->second.elements.end(), 1, std::multiplies<unsigned int>());
            unsigned int el = 0;
            while (getline(ss, token, ',')) {
                if (el == 0) {
                    if (!macro_env_init.emplace(std::string(key), std::vector<char>(elements * it->second.type_size)).second) {
                        THROW exception::TinyXMLError("Input file contains macro environment property '%s' multiple times, "
                            "in XMLStateReader::parse()\n", key);
                    }
                } else if (el >= elements) {
                    THROW exception::RapidJSONError("Input file contains macro environment property '%s' too many elements, expected %u,"
                        "in XMLStateReader::parse()\n", key, elements);
                }
                auto &mei = macro_env_init.at(key);
                if (val_type == std::type_index(typeid(float))) {
                    static_cast<float*>(static_cast<void*>(mei.data()))[el++] = stof(token);
                } else if (val_type == std::type_index(typeid(double))) {
                    static_cast<double*>(static_cast<void*>(mei.data()))[el++] = stod(token);
                } else if (val_type == std::type_index(typeid(int64_t))) {
                    static_cast<int64_t*>(static_cast<void*>(mei.data()))[el++] = stoll(token);
                } else if (val_type == std::type_index(typeid(uint64_t))) {
                    static_cast<uint64_t*>(static_cast<void*>(mei.data()))[el++] = stoull(token);
                } else if (val_type == std::type_index(typeid(int32_t))) {
                    static_cast<int32_t*>(static_cast<void*>(mei.data()))[el++] = static_cast<int32_t>(stoll(token));
                } else if (val_type == std::type_index(typeid(uint32_t))) {
                    static_cast<uint32_t*>(static_cast<void*>(mei.data()))[el++] = static_cast<uint32_t>(stoull(token));
                } else if (val_type == std::type_index(typeid(int16_t))) {
                    static_cast<int16_t*>(static_cast<void*>(mei.data()))[el++] = static_cast<int16_t>(stoll(token));
                } else if (val_type == std::type_index(typeid(uint16_t))) {
                    static_cast<uint16_t*>(static_cast<void*>(mei.data()))[el++] = static_cast<uint16_t>(stoull(token));
                } else if (val_type == std::type_index(typeid(int8_t))) {
                    static_cast<int8_t*>(static_cast<void*>(mei.data()))[el++] = static_cast<int8_t>(stoll(token));
                } else if (val_type == std::type_index(typeid(uint8_t))) {
                    static_cast<uint8_t*>(static_cast<void*>(mei.data()))[el++] = static_cast<uint8_t>(stoull(token));
                } else {
                    THROW exception::TinyXMLError("Model contains macro environment property '%s' of unsupported type '%s', "
                        "in XMLStateReader::parse()\n", key, val_type.name());
                }
            }
            if (el != elements && verbosity > Verbosity::Quiet) {
                fprintf(stderr, "Warning: Macro environment property '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                    key, elements, inputFile.c_str(), el);
            }
        }
    } else if (verbosity > Verbosity::Quiet) {
        fprintf(stderr, "Warning: Input file '%s' does not contain macro environment node.\n", inputFile.c_str());
    }

    // Count how many of each agent are in the file and resize state lists
    util::StringPairUnorderedMap<unsigned int> cts;
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        std::string agent_name = pElement->FirstChildElement("name")->GetText();
        tinyxml2::XMLElement *state_element = pElement->FirstChildElement("state");
        std::string state_name = state_element ? state_element->GetText() : getInitialState(model, agent_name);
        cts[{agent_name, state_name}]++;
    }

    // Init state lists to correct size
    for (const auto& it : cts) {
        const auto& agent = model->agents.find(it.first.first);
        if (agent == model->agents.end() || agent->second->states.find(it.first.second) == agent->second->states.end()) {
            THROW exception::InvalidAgentState("Agent '%s' with state '%s', found in input file '%s', is not part of the model description hierarchy, "
                "in XMLStateReader::parse()\n Ensure the input file is for the correct model.\n", it.first.first.c_str(), it.first.second.c_str(), inputFile.c_str());
        }
        auto [_it, _] = agents_map.emplace(it.first, std::make_shared<AgentVector>(*agent->second));
        _it->second->reserve(it.second);
    }

    bool hasWarnedElements = false;
    bool hasWarnedMissingVar = false;
    // Read in agent data
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        // Find agent name
        tinyxml2::XMLElement* pNameElement = pElement->FirstChildElement("name");
        const char* agentName = pNameElement->GetText();
        // Find agent state, use initial state if not set (means its old flame gpu 1 input file)
        tinyxml2::XMLElement* pStateElement = pElement->FirstChildElement("state");
        const std::string agentState = pStateElement ? std::string(pStateElement->GetText()) : getInitialState(model, agentName);
        const auto agentIt = agents_map.find({ agentName, agentState });
        if (agentIt == agents_map.end()) {
            THROW exception::InvalidAgentState("Agent '%s' with state '%s', found in input file '%s', is not part of the model description hierarchy, "
                "in XMLStateReader::parse()\n Ensure the input file is for the correct model.\n", agentName, agentState.c_str(), inputFile.c_str());
        }
        std::shared_ptr<AgentVector> &agentVec = agentIt->second;
        const VariableMap& agentVariables = agentVec->getVariableMetaData();
        // Create instance to store variable data in
        agentVec->push_back();
        AgentVector::Agent instance = agentVec->back();
        // Iterate agent variables
        for (auto iter = agentVariables.begin(); iter != agentVariables.end(); ++iter) {
            const std::string variable_name = iter->first;
            const auto &var_data = iter->second;
            tinyxml2::XMLElement* pVarElement = pElement->FirstChildElement(variable_name.c_str());
            if (pVarElement) {
                // Put string for the variable into a string stream
                std::stringstream ss(pVarElement->GetText());
                std::string token;
                const size_t v_size = var_data.type_size * var_data.elements;
                char* data = static_cast<char*>(const_cast<void*>(static_cast<std::shared_ptr<const AgentVector>>(agentVec)->data(variable_name)));
                // Iterate elements of the stringstream
                unsigned int el = 0;
                while (getline(ss, token, ',')) {
                    if (var_data.type == std::type_index(typeid(float))) {
                        const float t = stof(token);
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(double))) {
                        const double t = stod(token);
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(int64_t))) {
                        const int64_t t = stoll(token);
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(uint64_t))) {
                        const uint64_t t = stoull(token);
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(int32_t))) {
                        const int32_t t = static_cast<int32_t>(stoll(token));
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(uint32_t))) {
                        const uint32_t t = static_cast<uint32_t>(stoull(token));
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(int16_t))) {
                        const int16_t t = static_cast<int16_t>(stoll(token));
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(uint16_t))) {
                        const uint16_t t = static_cast<uint16_t>(stoull(token));
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(int8_t))) {
                        const int8_t t = static_cast<int8_t>(stoll(token));
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else if (var_data.type == std::type_index(typeid(uint8_t))) {
                        const uint8_t t = static_cast<uint8_t>(stoull(token));
                        memcpy(data + ((agentVec->size() - 1) * v_size) + (var_data.type_size * el++), &t, var_data.type_size);
                    } else {
                        THROW exception::TinyXMLError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                            "in XMLStateReader::parse()\n", agentName, variable_name.c_str(), var_data.type.name());
                    }
                }
                // Warn if var is wrong length
                if (el != var_data.elements && !hasWarnedElements && verbosity > Verbosity::Quiet) {
                    fprintf(stderr, "Warning: Agent '%s' variable '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                        agentName, variable_name.c_str(), var_data.elements, inputFile.c_str(), el);
                    hasWarnedElements = true;
                }
            } else if (!hasWarnedMissingVar && variable_name.find('_', 0) != 0 && verbosity > Verbosity::Quiet) {
                fprintf(stderr, "Warning: Agent '%s' variable '%s' is missing from, input file '%s'.\n",
                    agentName, variable_name.c_str(), inputFile.c_str());
                hasWarnedMissingVar = true;
            }
        }
    }

    // Mark input as loaded
    this->input_filepath = inputFile;
}

std::string XMLStateReader::getInitialState(const std::shared_ptr<const ModelData> &model, const std::string &agent_name) {
    const auto& it = model->agents.find(agent_name);
    if (it != model->agents.end()) {
        return it->second->initial_state;
    }
    return ModelData::DEFAULT_STATE;
}
}  // namespace io
}  // namespace flamegpu
