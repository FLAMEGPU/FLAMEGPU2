#include "flamegpu/io/XMLLogger.h"

#include <sstream>

#include "tinyxml2/tinyxml2.h"              // downloaded from https:// github.com/leethomason/tinyxml2, the list of xml parsers : http:// lars.ruoff.free.fr/xmlcpp/

#include "flamegpu/simulation/RunPlan.h"
#include "flamegpu/simulation/LogFrame.h"

namespace flamegpu {
namespace io {

#ifndef XMLCheckResult
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

XMLLogger::XMLLogger(const std::string &outPath, bool _prettyPrint, bool _truncateFile)
    : out_path(outPath)
    , prettyPrint(_prettyPrint)
    , truncateFile(_truncateFile) { }

void XMLLogger::log(const RunLog &log, const RunPlan &plan, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const {
  logCommon(log, &plan, false, logSteps, logExit, logStepTime, logExitTime);
}
void XMLLogger::log(const RunLog &log, bool logConfig, bool logSteps, bool logExit, bool logStepTime, bool logExitTime) const {
  logCommon(log, nullptr, logConfig, logSteps, logExit, logStepTime, logExitTime);
}

void XMLLogger::logCommon(const RunLog &log, const RunPlan *plan, bool doLogConfig, bool doLogSteps, bool doLogExit, bool doLogStepTime, bool doLogExitTime) const {
    tinyxml2::XMLDocument doc;

    tinyxml2::XMLNode * pRoot = doc.NewElement("log");
    doc.InsertFirstChild(pRoot);

    // Log config
    if (plan) {
        pRoot->InsertEndChild(logConfig(doc, *plan));
    } else if (doLogConfig) {
        pRoot->InsertEndChild(logConfig(doc, log));
    }

    // Log performance specs
    if (doLogStepTime || doLogExitTime) {
        pRoot->InsertEndChild(logPerformanceSpecs(doc, log));
    }

    // Log step log
    if (doLogSteps) {
        pRoot->InsertEndChild(logSteps(doc, log, doLogStepTime));
    }

    // Log exit log
    if (doLogExit) {
        pRoot->InsertEndChild(logExit(doc, log, doLogExitTime));
    }
    // export
    FILE *fptr = fopen(out_path.c_str(), truncateFile ? "w" : "a");
    if (fptr == nullptr) {
        THROW exception::TinyXMLError("Unable to open file '%s' for writing\n", out_path.c_str());
    }
    XMLCheckResult(doc.SaveFile(fptr, !prettyPrint));
    fwrite("\n", sizeof(char), 1, fptr);
    fclose(fptr);
}

tinyxml2::XMLNode *XMLLogger::logConfig(tinyxml2::XMLDocument &doc, const RunLog &log) const {
    tinyxml2::XMLElement *pConfigElement = doc.NewElement("config");
    {
        tinyxml2::XMLElement *pListElement;
        pListElement = doc.NewElement("random_seed");
        pListElement->SetText(log.getRandomSeed());
        pConfigElement->InsertEndChild(pListElement);
    }
    return pConfigElement;
}
tinyxml2::XMLNode *XMLLogger::logConfig(tinyxml2::XMLDocument &doc, const RunPlan &plan) const {
    tinyxml2::XMLElement *pConfigElement = doc.NewElement("config");
    {
        tinyxml2::XMLElement *pListElement;
        // Add static items
        pListElement = doc.NewElement("random_seed");
        pListElement->SetText(plan.getRandomSimulationSeed());
        pConfigElement->InsertEndChild(pListElement);
        pListElement = doc.NewElement("steps");
        pListElement->SetText(plan.getSteps());
        pConfigElement->InsertEndChild(pListElement);
        // Add dynamic environment overrides
        tinyxml2::XMLElement *pEnvElement = doc.NewElement("environment");
        {
            for (const auto &prop : plan.property_overrides) {
                const EnvironmentData::PropData &env_prop = plan.environment->at(prop.first);
                pListElement = doc.NewElement(prop.first.c_str());
                writeAny(pListElement, prop.second, env_prop.data.elements);
                pEnvElement->InsertEndChild(pListElement);
            }
        }
        pConfigElement->InsertEndChild(pEnvElement);
    }
    return pConfigElement;
}
tinyxml2::XMLNode* XMLLogger::logPerformanceSpecs(tinyxml2::XMLDocument& doc, const RunLog& log) const {
    tinyxml2::XMLElement* pConfigElement = doc.NewElement("performance_specs");
    {
        tinyxml2::XMLElement* pListElement;
        // Add static items
        pListElement = doc.NewElement("device_name");
        pListElement->SetText(log.getPerformanceSpecs().device_name.c_str());
        pConfigElement->InsertEndChild(pListElement);
        pListElement = doc.NewElement("device_cc_major");
        pListElement->SetText(log.getPerformanceSpecs().device_cc_major);
        pConfigElement->InsertEndChild(pListElement);
        pListElement = doc.NewElement("device_cc_minor");
        pListElement->SetText(log.getPerformanceSpecs().device_cc_minor);
        pConfigElement->InsertEndChild(pListElement);
        pListElement = doc.NewElement("cuda_version");
        pListElement->SetText(log.getPerformanceSpecs().cuda_version);
        pConfigElement->InsertEndChild(pListElement);
        pListElement = doc.NewElement("seatbelts");
        pListElement->SetText(log.getPerformanceSpecs().seatbelts);
        pConfigElement->InsertEndChild(pListElement);
        pListElement = doc.NewElement("flamegpu_version");
        pListElement->SetText(log.getPerformanceSpecs().flamegpu_version.c_str());
        pConfigElement->InsertEndChild(pListElement);
    }
    return pConfigElement;
}
tinyxml2::XMLNode *XMLLogger::logSteps(tinyxml2::XMLDocument &doc, const RunLog &log, bool logTime) const {
    tinyxml2::XMLElement *pStepsElement = doc.NewElement("steps");
    {
        for (const auto &step : log.getStepLog()) {
            pStepsElement->InsertEndChild(writeLogFrame(doc, step, logTime));
        }
    }
    return pStepsElement;
}
tinyxml2::XMLNode *XMLLogger::logExit(tinyxml2::XMLDocument &doc, const RunLog &log, bool logTime) const {
    return writeLogFrame(doc, log.getExitLog(), logTime);
}
tinyxml2::XMLNode* XMLLogger::writeLogFrame(tinyxml2::XMLDocument& doc, const StepLogFrame& frame, bool logTime) const {
    tinyxml2::XMLElement* pFrameElement = doc.NewElement("step");
    {
        if (logTime) {
            tinyxml2::XMLElement* pListElement;
            pListElement = doc.NewElement("step_time");
            pListElement->SetText(frame.getStepTime());
            pFrameElement->InsertEndChild(pListElement);
        }
        writeCommonLogFrame(doc, pFrameElement, frame);
    }
    return pFrameElement;
}
tinyxml2::XMLNode* XMLLogger::writeLogFrame(tinyxml2::XMLDocument & doc, const ExitLogFrame & frame, bool logTime) const {
    tinyxml2::XMLElement* pFrameElement = doc.NewElement("exit");
    {
        if (logTime) {
            tinyxml2::XMLElement* pListElement;
            pListElement = doc.NewElement("rtc_time");
            pListElement->SetText(frame.getRTCTime());
            pFrameElement->InsertEndChild(pListElement);
            pListElement = doc.NewElement("init_time");
            pListElement->SetText(frame.getInitTime());
            pFrameElement->InsertEndChild(pListElement);
            pListElement = doc.NewElement("exit_time");
            pListElement->SetText(frame.getExitTime());
            pFrameElement->InsertEndChild(pListElement);
            pListElement = doc.NewElement("total_time");
            pListElement->SetText(frame.getTotalTime());
            pFrameElement->InsertEndChild(pListElement);
        }
        writeCommonLogFrame(doc, pFrameElement, frame);
    }
    return pFrameElement;
}
void XMLLogger::writeCommonLogFrame(tinyxml2::XMLDocument &doc, tinyxml2::XMLElement* pFrameElement, const LogFrame & frame) const {
    tinyxml2::XMLElement *pListElement;
    // Add static items
    pListElement = doc.NewElement("step_index");
    pListElement->SetText(frame.getStepCount());
    pFrameElement->InsertEndChild(pListElement);
    // Add dynamic environment values
    if (frame.getEnvironment().size()) {
        tinyxml2::XMLElement *pEnvElement = doc.NewElement("environment");
        {
            for (const auto &prop : frame.getEnvironment()) {
                pListElement = doc.NewElement(prop.first.c_str());
                writeAny(pListElement, prop.second, prop.second.elements);
                pEnvElement->InsertEndChild(pListElement);
            }
        }
        pFrameElement->InsertEndChild(pEnvElement);
    }

    if (frame.getAgents().size()) {
        // Add dynamic agent values
        tinyxml2::XMLElement *pAgentsElement = doc.NewElement("agents");
        {
            // This assumes that sort order places all agents of same name, different state consecutively
            std::string current_agent;
            tinyxml2::XMLElement *pAgentsItemElement = nullptr;
            for (const auto &agent : frame.getAgents()) {
                // Start/end new agent
                if (current_agent != agent.first.first) {
                    if (!current_agent.empty())
                        pAgentsElement->InsertEndChild(pAgentsItemElement);
                    current_agent = agent.first.first;
                    pAgentsItemElement = doc.NewElement(current_agent.c_str());
                }
                // Start new state
                tinyxml2::XMLElement *pStateElement = doc.NewElement(agent.first.second.c_str());
                {
                    // Log agent count if provided
                    if (agent.second.second != UINT_MAX) {
                        tinyxml2::XMLElement *pCountElement = doc.NewElement("count");
                        pCountElement->SetText(agent.second.second);
                        pStateElement->InsertEndChild(pCountElement);
                    }
                    if (agent.second.first.size()) {
                        tinyxml2::XMLElement *pVariablesBlock = doc.NewElement("variables");
                        // This assumes that sort order places all variables of same name, different reduction consecutively
                        std::string current_variable;
                        tinyxml2::XMLElement *pVariableElement = nullptr;
                        // Log each reduction
                        for (auto &var : agent.second.first) {
                            // Start/end new variable
                            if (current_variable != var.first.name) {
                                if (!current_variable.empty())
                                    pVariablesBlock->InsertEndChild(pVariableElement);
                                current_variable = var.first.name;
                                pVariableElement = doc.NewElement(current_variable.c_str());
                            }
                            // Build name key for the variable & log value
                            tinyxml2::XMLElement *pValueElement = doc.NewElement(LoggingConfig::toString(var.first.reduction));
                            writeAny(pValueElement, var.second, 1);
                            pVariableElement->InsertEndChild(pValueElement);
                        }
                        if (!current_variable.empty())
                            pVariablesBlock->InsertEndChild(pVariableElement);
                        pStateElement->InsertEndChild(pVariablesBlock);
                    }
                }
                pAgentsItemElement->InsertEndChild(pStateElement);
            }
            if (!current_agent.empty())
                pAgentsElement->InsertEndChild(pAgentsItemElement);
        }
        pFrameElement->InsertEndChild(pAgentsElement);
    }
}

void XMLLogger::writeAny(tinyxml2::XMLElement *pElement, const detail::Any &value, const unsigned int elements) const {
    std::stringstream ss;
    // Loop through elements, to construct csv string
    for (unsigned int el = 0; el < elements; ++el) {
        if (value.type == std::type_index(typeid(float))) {
            ss << static_cast<const float*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(double))) {
             ss << static_cast<const double*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(int64_t))) {
            ss << static_cast<const int64_t*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(uint64_t))) {
             ss << static_cast<const uint64_t*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(int32_t))) {
            ss << static_cast<const int32_t*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(uint32_t))) {
             ss << static_cast<const uint32_t*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(int16_t))) {
             ss << static_cast<const int16_t*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(uint16_t))) {
             ss << static_cast<const uint16_t*>(value.ptr)[el];
        } else if (value.type == std::type_index(typeid(int8_t))) {
            ss << static_cast<int32_t>(static_cast<const int8_t*>(value.ptr)[el]);  // Char outputs weird if being used as an integer
        } else if (value.type == std::type_index(typeid(uint8_t))) {
            ss << static_cast<uint32_t>(static_cast<const uint8_t*>(value.ptr)[el]);  // Char outputs weird if being used as an integer
        } else if (value.type == std::type_index(typeid(char))) {
            ss << static_cast<int32_t>(static_cast<const char*>(value.ptr)[el]);  // Char outputs weird if being used as an integer
        } else {
            THROW exception::TinyXMLError("Attempting to export value of unsupported type '%s', "
                "in XMLLogger::writeAny()\n", value.type.name());
       }
        if (el + 1 != elements)
            ss << ",";
    }
    pElement->SetText(ss.str().c_str());
}

}  // namespace io
}  // namespace flamegpu
