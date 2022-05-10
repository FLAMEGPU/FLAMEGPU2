
/**
 * @file
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#include "flamegpu/io/XMLStateReader.h"
#include <sstream>
#include <algorithm>
#include <tuple>
#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/gpu/CUDASimulation.h"

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
        throw exception::InvalidInputFile("TinyXML error: File could not be opened.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_FILE_READ_ERROR : \
        throw exception::InvalidInputFile("TinyXML error: File could not be read.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_PARSING_ELEMENT : \
    case tinyxml2::XML_ERROR_PARSING_ATTRIBUTE : \
    case tinyxml2::XML_ERROR_PARSING_TEXT : \
    case tinyxml2::XML_ERROR_PARSING_CDATA : \
    case tinyxml2::XML_ERROR_PARSING_COMMENT : \
    case tinyxml2::XML_ERROR_PARSING_DECLARATION : \
    case tinyxml2::XML_ERROR_PARSING_UNKNOWN : \
    case tinyxml2::XML_ERROR_PARSING : \
        throw exception::TinyXMLError("TinyXML error: Error parsing file.\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_EMPTY_DOCUMENT : \
        throw exception::TinyXMLError("TinyXML error: XML_ERROR_EMPTY_DOCUMENT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_MISMATCHED_ELEMENT : \
        throw exception::TinyXMLError("TinyXML error: XML_ERROR_MISMATCHED_ELEMENT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_CAN_NOT_CONVERT_TEXT : \
        throw exception::TinyXMLError("TinyXML error: XML_CAN_NOT_CONVERT_TEXT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_NO_TEXT_NODE : \
        throw exception::TinyXMLError("TinyXML error: XML_NO_TEXT_NODE\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ELEMENT_DEPTH_EXCEEDED : \
        throw exception::TinyXMLError("TinyXML error: XML_ELEMENT_DEPTH_EXCEEDED\n Error code: %d", a_eResult); \
    case tinyxml2::XML_ERROR_COUNT : \
        throw exception::TinyXMLError("TinyXML error: XML_ERROR_COUNT\n Error code: %d", a_eResult); \
    case tinyxml2::XML_NO_ATTRIBUTE: \
        throw exception::TinyXMLError("TinyXML error: XML_NO_ATTRIBUTE\n Error code: %d", a_eResult); \
    case tinyxml2::XML_WRONG_ATTRIBUTE_TYPE : \
        throw exception::TinyXMLError("TinyXML error: XML_WRONG_ATTRIBUTE_TYPE\n Error code: %d", a_eResult); \
    default: \
        throw exception::TinyXMLError("TinyXML error: Unrecognised error code\n Error code: %d", a_eResult); \
    } \
}
#endif

XMLStateReader::XMLStateReader(
    const std::string &model_name,
    const std::unordered_map<std::string, EnvironmentDescription::PropData> &env_desc,
    std::unordered_map<std::string, util::Any>&env_init,
    util::StringPairUnorderedMap<std::shared_ptr<AgentVector>> &model_state,
    const std::string &input,
    Simulation *sim_instance)
    : StateReader(model_name, env_desc, env_init, model_state, input, sim_instance) {}

std::string XMLStateReader::getInitialState(const std::string &agent_name) const {
    for (const auto &i : model_state) {
        if (agent_name == i.first.first)
            return i.second->getInitialState();
    }
    return ModelData::DEFAULT_STATE;
}
/**
* \brief parses the xml file
*/
int XMLStateReader::parse() {
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
        if (sim_instance) {
            tinyxml2::XMLElement *pSimCfgBlock = pElement->FirstChildElement("simulation");
            for (auto simCfgElement = pSimCfgBlock->FirstChildElement(); simCfgElement; simCfgElement = simCfgElement->NextSiblingElement()) {
                std::string key = simCfgElement->Value();
                std::string val = simCfgElement->GetText() ? simCfgElement->GetText() : "";
                if (key == "input_file") {
                    if (inputFile != val && !val.empty())
                        printf("Warning: Input file '%s' refers to second input file '%s', this will not be loaded.\n", inputFile.c_str(), val.c_str());
                    // sim_instance->SimulationConfig().input_file = val;
                } else if (key == "step_log_file") {
                    sim_instance->SimulationConfig().step_log_file = val;
                } else if (key == "exit_log_file") {
                    sim_instance->SimulationConfig().exit_log_file = val;
                } else if (key == "common_log_file") {
                    sim_instance->SimulationConfig().common_log_file = val;
                } else if (key == "truncate_log_files") {
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        sim_instance->SimulationConfig().truncate_log_files = true;
                    } else if (val == "false") {
                        sim_instance->SimulationConfig().truncate_log_files = false;
                    } else {
                        sim_instance->SimulationConfig().truncate_log_files = static_cast<bool>(stoll(val));
                    }
                } else if (key == "random_seed") {
                    sim_instance->SimulationConfig().random_seed = static_cast<uint64_t>(stoull(val));
                } else if (key == "steps") {
                    sim_instance->SimulationConfig().steps = static_cast<unsigned int>(stoull(val));
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
                } else if (key == "console_mode") {
#ifdef VISUALISATION
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        sim_instance->SimulationConfig().console_mode = true;
                    } else if (val == "false") {
                        sim_instance->SimulationConfig().console_mode = false;
                    } else {
                        sim_instance->SimulationConfig().console_mode = static_cast<bool>(stoll(val));
                    }
#else
                    if (val == "false") {
                        fprintf(stderr, "Warning: Cannot disable 'console_mode' with input file '%s', FLAMEGPU2 library has not been built with visualisation support enabled.\n", inputFile.c_str());
                    }
#endif
                }  else {
                    fprintf(stderr, "Warning: Input file '%s' contains unexpected simulation config property '%s'.\n", inputFile.c_str(), key.c_str());
                }
            }
        }
        // CUDA config
        CUDASimulation *cudamodel_instance = dynamic_cast<CUDASimulation*>(sim_instance);
        if (cudamodel_instance) {
            tinyxml2::XMLElement *pCUDACfgBlock = pElement->FirstChildElement("cuda");
            for (auto cudaCfgElement = pCUDACfgBlock->FirstChildElement(); cudaCfgElement; cudaCfgElement = cudaCfgElement->NextSiblingElement()) {
                std::string key = cudaCfgElement->Value();
                std::string val = cudaCfgElement->GetText();
                if (key == "device_id") {
                    cudamodel_instance->CUDAConfig().device_id = static_cast<unsigned int>(stoull(val));
                } else if (key == "inLayerConcurrency") {
                    for (auto& c : val)
                        c = static_cast<char>(::tolower(c));
                    if (val == "true") {
                        cudamodel_instance->CUDAConfig().inLayerConcurrency = true;
                    } else if (val == "false") {
                        cudamodel_instance->CUDAConfig().inLayerConcurrency = false;
                    } else {
                        cudamodel_instance->CUDAConfig().inLayerConcurrency = static_cast<bool>(stoll(val));
                    }
                } else {
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
        for (auto envElement = pElement->FirstChildElement(); envElement; envElement = envElement->NextSiblingElement()) {
            const char *key = envElement->Value();
            std::stringstream ss(envElement->GetText());
            std::string token;
            const auto it = env_desc.find(std::string(key));
            if (it == env_desc.end()) {
                THROW exception::TinyXMLError("Input file contains unrecognised environment property '%s',"
                    "in XMLStateReader::parse()\n", key);
            }
            const std::type_index val_type = it->second.data.type;
            const auto elements = it->second.data.elements;
            unsigned int el = 0;
            while (getline(ss, token, ',')) {
                if (el == 0) {
                    if (!env_init.emplace(std::string(key), util::Any(it->second.data)).second) {
                        THROW exception::TinyXMLError("Input file contains environment property '%s' multiple times, "
                            "in XMLStateReader::parse()\n", key);
                    }
                } else if (el >= it->second.data.elements) {
                    THROW exception::RapidJSONError("Input file contains environment property '%s' with %u elements expected %u,"
                        "in XMLStateReader::parse()\n", key, el, it->second.data.elements);
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
            if (el != elements) {
                fprintf(stderr, "Warning: Environment array property '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                    key, elements, inputFile.c_str(), el);
            }
        }
    } else {
        fprintf(stderr, "Warning: Input file '%s' does not contain environment node.\n", inputFile.c_str());
    }

    // Count how many of each agent are in the file and resize state lists
    util::StringPairUnorderedMap<unsigned int> cts;
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        std::string agent_name = pElement->FirstChildElement("name")->GetText();
        tinyxml2::XMLElement *state_element = pElement->FirstChildElement("state");
        std::string state_name = state_element ? state_element->GetText() : getInitialState(agent_name);
        cts[{agent_name, state_name}]++;
    }
    // Resize state lists (greedy, all lists are resized to max size of state)
    for (auto& it : model_state) {
        auto f = cts.find(it.first);
        if (f != cts.end())
          it.second->reserve(f->second);
    }

    // Read in agent data
    for (pElement = pRoot->FirstChildElement("xagent"); pElement != nullptr; pElement = pElement->NextSiblingElement("xagent")) {
        // Find agent name
        tinyxml2::XMLElement* pNameElement = pElement->FirstChildElement("name");
        const char* agentName = pNameElement->GetText();
        // Find agent state, use initial state if not set (means its old flame gpu 1 input file)
        tinyxml2::XMLElement* pStateElement = pElement->FirstChildElement("state");
        const std::string agentState = pStateElement ? std::string(pStateElement->GetText()) : getInitialState(agentName);
        const auto agentIt = model_state.find({ agentName, agentState });
        if (agentIt == model_state.end()) {
            THROW exception::InvalidAgentState("Agent '%s' with state '%s', found in input file '%s', is not part of the model description hierarchy, "
                "in XMLStateReader::parse()\n Ensure the input file is for the correct model.\n", agentName, agentState.c_str(), inputFile.c_str());
        }
        std::shared_ptr<AgentVector> &agentVec = agentIt->second;
        const VariableMap& agentVariables = agentVec->getVariableMetaData();
        // Create instance to store variable data in
        agentVec->push_back();
        AgentVector::Agent instance = agentVec->back();
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
                if (el != var_data.elements && !hasWarnedElements) {
                    fprintf(stderr, "Warning: Agent '%s' variable '%s' expects '%u' elements, input file '%s' contains '%u' elements.\n",
                        agentName, variable_name.c_str(), var_data.elements, inputFile.c_str(), el);
                    hasWarnedElements = true;
                }
            } else if (!hasWarnedMissingVar && variable_name.find('_', 0) != 0) {
                fprintf(stderr, "Warning: Agent '%s' variable '%s' is missing from, input file '%s'.\n",
                    agentName, variable_name.c_str(), inputFile.c_str());
                hasWarnedMissingVar = true;
            }
        }
    }

    return tinyxml2::XML_SUCCESS;
}

}  // namespace io
}  // namespace flamegpu
