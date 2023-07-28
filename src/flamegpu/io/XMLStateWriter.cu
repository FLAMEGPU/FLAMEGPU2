#include "flamegpu/io/XMLStateWriter.h"

#include <numeric>
#include <sstream>
#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/simulation/AgentVector.h"
#include "flamegpu/simulation/detail/EnvironmentManager.cuh"

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

XMLStateWriter::XMLStateWriter()
    : StateWriter() {}
void XMLStateWriter::beginWrite(const std::string &output_file, bool pretty_print) {
    this->outputPath = output_file;
    this->prettyPrint = pretty_print;
    if (doc || pRoot) {
        THROW exception::UnknownInternalError("Writing already active, in XMLStateWriter::beginWrite()");
    }
    doc = std::make_unique<tinyxml2::XMLDocument>();
    // Begin Json file
    pRoot = doc->NewElement("states");
    doc->InsertFirstChild(pRoot);

    // Clear flags
    this->config_written = false;
    this->stats_written = false;
    this->environment_written = false;
    this->macro_environment_written = false;
    this->agents_written = false;
}
void XMLStateWriter::endWrite() {
    if (!doc || !pRoot) {
        THROW exception::UnknownInternalError("Writing not active, in XMLStateWriter::endWrite()");
    }

    // End Json file
    tinyxml2::XMLError errorId = doc->SaveFile(outputPath.c_str(), !prettyPrint);
    XMLCheckResult(errorId);

    pRoot = nullptr;
    doc.reset();
}


void XMLStateWriter::writeConfig(const Simulation *sim_instance) {
    if (!doc || !pRoot) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeConfig(), in XMLStateWriter::writeConfig()");
    } else if (config_written) {
        THROW exception::UnknownInternalError("writeConfig() can only be called once per write session, in XMLStateWriter::writeConfig()");
    }

    // Output config elements
    tinyxml2::XMLElement *pElement = doc->NewElement("config");
    {
        // Sim config
        tinyxml2::XMLElement *pSimCfg = doc->NewElement("simulation");
        {
            const auto &sim_cfg = sim_instance->getSimulationConfig();
            tinyxml2::XMLElement *pListElement = nullptr;
            // Input file
            pListElement = doc->NewElement("input_file");
            pListElement->SetText(sim_cfg.input_file.c_str());
            pSimCfg->InsertEndChild(pListElement);
            // Step log file
            pListElement = doc->NewElement("step_log_file");
            pListElement->SetText(sim_cfg.step_log_file.c_str());
            pSimCfg->InsertEndChild(pListElement);
            // Exit log file
            pListElement = doc->NewElement("exit_log_file");
            pListElement->SetText(sim_cfg.exit_log_file.c_str());
            pSimCfg->InsertEndChild(pListElement);
            // Common log file
            pListElement = doc->NewElement("common_log_file");
            pListElement->SetText(sim_cfg.common_log_file.c_str());
            pSimCfg->InsertEndChild(pListElement);
            // Truncate log files
            pListElement = doc->NewElement("truncate_log_files");
            pListElement->SetText(sim_cfg.truncate_log_files);
            pSimCfg->InsertEndChild(pListElement);
            // Random seed
            pListElement = doc->NewElement("random_seed");
            pListElement->SetText(sim_cfg.random_seed);
            pSimCfg->InsertEndChild(pListElement);
            // Steps
            pListElement = doc->NewElement("steps");
            pListElement->SetText(sim_cfg.steps);
            pSimCfg->InsertEndChild(pListElement);
            // Verbose output
            pListElement = doc->NewElement("verbosity");
            pListElement->SetText(static_cast<unsigned int>(sim_cfg.verbosity));
            pSimCfg->InsertEndChild(pListElement);
            // Timing Output
            pListElement = doc->NewElement("timing");
            pListElement->SetText(sim_cfg.timing);
            pSimCfg->InsertEndChild(pListElement);
#ifdef FLAMEGPU_VISUALISATION
            // Console Mode
            pListElement = doc->NewElement("console_mode");
            pListElement->SetText(sim_cfg.console_mode);
            pSimCfg->InsertEndChild(pListElement);
#endif
        }
        pElement->InsertEndChild(pSimCfg);

        // Cuda config
        if (auto *cudamodel_instance = dynamic_cast<const CUDASimulation*>(sim_instance)) {
            tinyxml2::XMLElement *pCUDACfg = doc->NewElement("cuda");
            {
                const auto &cuda_cfg = cudamodel_instance->getCUDAConfig();
                tinyxml2::XMLElement *pListElement = nullptr;
                // Device ID
                pListElement = doc->NewElement("device_id");
                pListElement->SetText(cuda_cfg.device_id);
                pCUDACfg->InsertEndChild(pListElement);
                // inLayerConcurrency
                pListElement = doc->NewElement("inLayerConcurrency");
                pListElement->SetText(cuda_cfg.inLayerConcurrency);
                pCUDACfg->InsertEndChild(pListElement);
            }
            pElement->InsertEndChild(pCUDACfg);
        }
    }
    pRoot->InsertEndChild(pElement);

    config_written = true;
}
void XMLStateWriter::writeStats(unsigned int iterations) {
    if (!doc || !pRoot) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeStats(), in XMLStateWriter::writeStats()");
    } else if (stats_written) {
        THROW exception::UnknownInternalError("writeStats() can only be called once per write session, in XMLStateWriter::writeStats()");
    }

    // Redundant for FLAMEGPU1 backwards compatibility
    tinyxml2::XMLElement *pElement = doc->NewElement("itno");
    pElement->SetText(iterations);
    pRoot->InsertEndChild(pElement);

    // Output stats elements
    pElement = doc->NewElement("stats");
    {
        tinyxml2::XMLElement *pListElement = nullptr;
        // Input file
        pListElement = doc->NewElement("step_count");
        pListElement->SetText(iterations);
        pElement->InsertEndChild(pListElement);
    }
    pRoot->InsertEndChild(pElement);

    stats_written = true;
}

void XMLStateWriter::writeEnvironment(const std::shared_ptr<const detail::EnvironmentManager>& env_manager) {
    if (!doc || !pRoot) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeEnvironment(), in XMLStateWriter::writeEnvironment()");
    } else if (environment_written) {
        THROW exception::UnknownInternalError("writeEnvironment() can only be called once per write session, in XMLStateWriter::writeEnvironment()");
    }

    tinyxml2::XMLElement *pElement = doc->NewElement("environment");
    if (env_manager) {
        const char* env_buffer = reinterpret_cast<const char*>(env_manager->getHostBuffer());
        // for each environment property
        for (auto &a : env_manager->getPropertiesMap()) {
            tinyxml2::XMLElement* pListElement = doc->NewElement(a.first.c_str());
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
                    THROW exception::TinyXMLError("Model contains environment property '%s' of unsupported type '%s', "
                        "in XMLStateWriter::writeEnvironment()\n", a.first.c_str(), a.second.type.name());
                }
                if (el + 1 != a.second.elements)
                    ss << ",";
            }
            pListElement->SetText(ss.str().c_str());
            pElement->InsertEndChild(pListElement);
        }
    }
    pRoot->InsertEndChild(pElement);

    environment_written = true;
}
void XMLStateWriter::writeMacroEnvironment(const std::shared_ptr<const detail::CUDAMacroEnvironment>& macro_env, std::initializer_list<std::string> filter) {
    if (!doc || !pRoot) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeMacroEnvironment(), in XMLStateWriter::writeMacroEnvironment()");
    } else if (macro_environment_written) {
        THROW exception::UnknownInternalError("writeMacroEnvironment() can only be called once per write session, in XMLStateWriter::writeMacroEnvironment()");
    }

    tinyxml2::XMLElement *pElement = doc->NewElement("macro_environment");
    if (macro_env) {
        const std::map<std::string, detail::CUDAMacroEnvironment::MacroEnvProp>& m_properties = macro_env->getPropertiesMap();
        for (const auto &_filter : filter) {
            if (m_properties.find(_filter) == m_properties.end()) {
                THROW exception::InvalidEnvProperty("Macro property '%s' specified in filter does not exist, in XMLStateWriter::writeMacroEnvironment()", _filter.c_str());
            }
        }
        std::set<std::string> filter_set = filter;
        // Calculate largest buffer in map
        size_t max_len = 0;
        for (const auto& [_, prop] : m_properties) {
            max_len = std::max(max_len, std::accumulate(prop.elements.begin(), prop.elements.end(), 1, std::multiplies<unsigned int>()) * prop.type_size);
        }
        if (max_len) {
            // Allocate temp buffer
            char* const t_buffer = static_cast<char*>(malloc(max_len));
            // Write out each array (all are written out as 1D arrays for simplicity given variable dimensions)
            for (const auto& [name, prop] : m_properties) {
                if (!filter_set.empty() && filter_set.find(name) == filter_set.end())
                    continue;
                // Copy data
                const size_t element_ct = std::accumulate(prop.elements.begin(), prop.elements.end(), 1, std::multiplies<unsigned int>());
                gpuErrchk(cudaMemcpy(t_buffer, prop.d_ptr, element_ct * prop.type_size, cudaMemcpyDeviceToHost));

                tinyxml2::XMLElement* pListElement = doc->NewElement(name.c_str());
                pListElement->SetAttribute("type", prop.type.name());

                // Loop through dimensions to construct dimensions string
                // Clip trailing 1 dimensions
                std::stringstream ss;
                size_t sum = 1;
                for (size_t j = 0; j < prop.elements.size(); ++j) {
                    ss << prop.elements[j];
                    sum *= prop.elements[j];
                    if (sum == element_ct)
                        break;
                    ss << ",";
                }
                pListElement->SetAttribute("dimensions", ss.str().c_str());
                ss.str("");
                ss.clear();

                // Output elements
                // Loop through elements, to construct csv string
                for (size_t i = 0; i < element_ct; ++i) {
                    if (prop.type == std::type_index(typeid(float))) {
                        ss << *reinterpret_cast<const float*>(t_buffer + i * sizeof(float));
                    } else if (prop.type == std::type_index(typeid(double))) {
                        ss << *reinterpret_cast<const double*>(t_buffer + i * sizeof(double));
                    } else if (prop.type == std::type_index(typeid(int64_t))) {
                        ss << *reinterpret_cast<const int64_t*>(t_buffer + i * sizeof(int64_t));
                    } else if (prop.type == std::type_index(typeid(uint64_t))) {
                        ss << *reinterpret_cast<const uint64_t*>(t_buffer + i * sizeof(uint64_t));
                    } else if (prop.type == std::type_index(typeid(int32_t))) {
                        ss << *reinterpret_cast<const int32_t*>(t_buffer + i * sizeof(int32_t));
                    } else if (prop.type == std::type_index(typeid(uint32_t))) {
                        ss << *reinterpret_cast<const uint32_t*>(t_buffer + i * sizeof(uint32_t));
                    } else if (prop.type == std::type_index(typeid(int16_t))) {
                        ss << *reinterpret_cast<const int16_t*>(t_buffer + i * sizeof(int16_t));
                    } else if (prop.type == std::type_index(typeid(uint16_t))) {
                        ss << *reinterpret_cast<const uint16_t*>(t_buffer + i * sizeof(uint16_t));
                    } else if (prop.type == std::type_index(typeid(int8_t))) {
                        ss << static_cast<int32_t>(*reinterpret_cast<const int8_t*>(t_buffer + i * sizeof(int8_t)));  // Char outputs weird if being used as an integer
                    } else if (prop.type == std::type_index(typeid(uint8_t))) {
                        ss << static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(t_buffer + i * sizeof(uint8_t)));  // Char outputs weird if being used as an integer
                    } else {
                        THROW exception::TinyXMLError("Model contains macro environment property '%s' of unsupported type '%s', "
                            "in XMLStateWriter::writeMacroEnvironment()\n", name.c_str(), prop.type.name());
                    }
                    if (i + 1 != element_ct)
                        ss << ",";
                }
                pListElement->SetText(ss.str().c_str());
                pElement->InsertEndChild(pListElement);
            }
            // Release temp buffer
            free(t_buffer);
        }
    }
    pRoot->InsertEndChild(pElement);

    macro_environment_written = true;
}
void XMLStateWriter::writeAgents(const util::StringPairUnorderedMap<std::shared_ptr<const AgentVector>>& agents_map) {
    if (!doc || !pRoot) {
        THROW exception::UnknownInternalError("beginWrite() must be called before writeAgents(), in XMLStateWriter::writeAgents()");
    } else if (agents_written) {
        THROW exception::UnknownInternalError("writeAgents() can only be called once per write session, in XMLStateWriter::writeAgents()");
    }

    // for each agent types
    for (const auto &[key, vec] : agents_map) {
        // For each agent state
        const std::string &agent_name = key.first;
        const std::string &state_name = key.second;

        unsigned int populationSize = vec->size();
        if (populationSize) {
            for (unsigned int i = 0; i < populationSize; ++i) {
                // Create vars block
                tinyxml2::XMLElement * pXagentElement = doc->NewElement("xagent");

                const AgentVector::CAgent instance = vec->at(i);
                const VariableMap &mm = vec->getVariableMetaData();

                // Add agent's name to block
                tinyxml2::XMLElement * pXagentNameElement = doc->NewElement("name");
                pXagentNameElement->SetText(agent_name.c_str());
                pXagentElement->InsertEndChild(pXagentNameElement);
                // Add state's name to block
                tinyxml2::XMLElement * pStateNameElement = doc->NewElement("state");
                pStateNameElement->SetText(state_name.c_str());
                pXagentElement->InsertEndChild(pStateNameElement);

                // for each variable
                for (auto iter_mm = mm.begin(); iter_mm != mm.end(); ++iter_mm) {
                    const std::string variable_name = iter_mm->first;

                    tinyxml2::XMLElement* pListElement = doc->NewElement(variable_name.c_str());
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
                            THROW exception::TinyXMLError("Agent '%s' contains variable '%s' of unsupported type '%s', "
                                "in XMLStateWriter::writeFullModelState()\n", agent_name.c_str(), variable_name.c_str(), iter_mm->second.type.name());
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

    agents_written = true;
}
}  // namespace io
}  // namespace flamegpu
