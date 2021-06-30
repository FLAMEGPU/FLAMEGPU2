#include <nvrtc.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <regex>

#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/util/detail/cxxname.hpp"

namespace flamegpu {


/**
 * Constructors
 */

AgentFunctionDescription::AgentFunctionDescription(const std::shared_ptr<const ModelData> &_model, AgentFunctionData *const description)
    : model(_model)
    , function(description) { }

bool AgentFunctionDescription::operator==(const AgentFunctionDescription& rhs) const {
    return *this->function == *rhs.function;  // Compare content is functionally the same
}
bool AgentFunctionDescription::operator!=(const AgentFunctionDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Accessors
 */
void AgentFunctionDescription::setInitialState(const std::string &init_state) {
    if (auto p = function->parent.lock()) {
        if (p->description->hasState(init_state)) {
            // Check if this agent function is already in a layer
            auto mdl = model.lock();
            if (!mdl) {
                THROW exception::ExpiredWeakPtr();
            }
            for (const auto &l : mdl->layers) {
                for (const auto &f : l->agent_functions) {
                    // Agent fn is in layer
                    if (f->name == this->function->name) {
                        // search all functions in that layer
                        for (const auto &f2 : l->agent_functions) {
                            if (const auto &a2 = f2->parent.lock()) {
                                if (const auto &a1 = this->function->parent.lock()) {
                                    // Same agent
                                    if (a2->name == a1->name) {
                                        // Skip ourself
                                        if (f2->name == this->function->name)
                                            continue;
                                        if (f2->initial_state == init_state ||
                                            f2->end_state == init_state) {
                                            THROW exception::InvalidAgentFunc("Agent functions's '%s' and '%s', within the same layer "
                                                "cannot share any input or output states, this is not permitted, "
                                                "in AgentFunctionDescription::setInitialState().",
                                                f2->name.c_str(), this->function->name.c_str());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Checks passed, make change
            this->function->initial_state = init_state;
        } else {
            THROW exception::InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setInitialState().",
                p->name.c_str(), init_state.c_str());
        }
    } else {
        THROW exception::InvalidParent("Agent parent has expired, "
            "in AgentFunctionDescription::setInitialState().");
    }
}
void AgentFunctionDescription::setEndState(const std::string &exit_state) {
    if (auto p = function->parent.lock()) {
        if (p->description->hasState(exit_state)) {
            // Check if this agent function is already in a layer
            auto mdl = model.lock();
            if (!mdl) {
                THROW exception::ExpiredWeakPtr();
            }
            for (const auto &l : mdl->layers) {
                for (const auto &f : l->agent_functions) {
                    // Agent fn is in layer
                    if (f->name == this->function->name) {
                        // search all functions in that layer
                        for (const auto &f2 : l->agent_functions) {
                            if (const auto &a2 = f2->parent.lock()) {
                                if (const auto &a1 = this->function->parent.lock()) {
                                    // Same agent
                                    if (a2->name == a1->name) {
                                        // Skip ourself
                                        if (f2->name == this->function->name)
                                            continue;
                                        if (f2->initial_state == exit_state ||
                                            f2->end_state == exit_state) {
                                            THROW exception::InvalidAgentFunc("Agent functions's '%s' and '%s', within the same layer "
                                                "cannot share any input or output states, this is not permitted, "
                                                "in AgentFunctionDescription::setEndState().",
                                                f2->name.c_str(), this->function->name.c_str());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Checks passed, make change
            this->function->end_state = exit_state;
        } else {
            THROW exception::InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setEndState().",
                p->name.c_str(), exit_state.c_str());
        }
    } else {
        THROW exception::InvalidParent("Agent parent has expired, "
            "in AgentFunctionDescription::setEndState().");
    }
}
void AgentFunctionDescription::setMessageInput(const std::string &message_name) {
    if (auto other = function->message_output.lock()) {
        if (message_name == other->name) {
            THROW exception::InvalidMessageName("Message '%s' is already bound as message output in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageInput().",
                message_name.c_str(), function->name.c_str());
        }
    }
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->messages.find(message_name);
    if (a != mdl->messages.end()) {
        // Just compare the classname is the same, to allow for the various approaches to namespace use. This should only be required for RTC functions.
        auto msg_in_classname = util::detail::cxxname::getUnqualifiedName(this->function->msg_in_type);
        auto demangledClassName = util::detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(a->second->getType()));
        if (msg_in_classname == demangledClassName) {
            this->function->message_input = a->second;
        } else {
            THROW exception::InvalidMessageType("Message ('%s') type '%s' does not match type '%s' applied to FLAMEGPU_AGENT_FUNCTION ('%s'), "
                "in AgentFunctionDescription::setMessageInput().",
                message_name.c_str(), demangledClassName.c_str(), msg_in_classname.c_str(), this->function->name.c_str());
        }
    } else {
        THROW exception::InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageInput().",
            mdl->name.c_str(), message_name.c_str());
    }
}
void AgentFunctionDescription::setMessageInput(MsgBruteForce::Description &message) {
    if (message.model.lock() != function->description->model.lock()) {
        THROW exception::DifferentModel("Attempted to use agent description from a different model, "
            "in AgentFunctionDescription::setAgentOutput().");
    }
    if (auto other = function->message_output.lock()) {
        if (message.getName() == other->name) {
            THROW exception::InvalidMessageName("Message '%s' is already bound as message output in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageInput().",
                message.getName().c_str(), function->name.c_str());
        }
    }
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->messages.find(message.getName());
    if (a != mdl->messages.end()) {
        if (a->second->description.get() == &message) {
            // Just compare the classname is the same, to allow for the various approaches to namespace use. This should only be required for RTC functions.
            auto msg_in_classname = util::detail::cxxname::getUnqualifiedName(this->function->msg_in_type);
            auto demangledClassName = util::detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(a->second->getType()));
            if (msg_in_classname == demangledClassName) {
                this->function->message_input = a->second;
            } else {
                THROW exception::InvalidMessageType("Message ('%s') type '%s' does not match type '%s' applied to FLAMEGPU_AGENT_FUNCTION ('%s'), "
                    "in AgentFunctionDescription::setMessageInput().",
                    a->second->name.c_str(), demangledClassName.c_str(), msg_in_classname.c_str(), this->function->name.c_str());
            }
        } else {
            THROW exception::InvalidMessage("Message '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setMessageInput().",
                message.getName().c_str(), mdl->name.c_str());
        }
    } else {
        THROW exception::InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageInput().",
            mdl->name.c_str(), message.getName().c_str());
    }
}
void AgentFunctionDescription::setMessageOutput(const std::string &message_name) {
    if (auto other = function->message_input.lock()) {
        if (message_name == other->name) {
            THROW exception::InvalidMessageName("Message '%s' is already bound as message output in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageOutput().",
                message_name.c_str(), function->name.c_str());
        }
    }
    // Clear old value
    if (this->function->message_output_optional) {
        if (auto b = this->function->message_output.lock()) {
            b->optional_outputs--;
        }
    }
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->messages.find(message_name);
    if (a != mdl->messages.end()) {
        // Just compare the classname is the same, to allow for the various approaches to namespace use. This should only be required for RTC functions.
        auto msg_out_classname = util::detail::cxxname::getUnqualifiedName(this->function->msg_out_type);
        auto demangledClassName = util::detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(a->second->getType()));
        if (msg_out_classname == demangledClassName) {
            this->function->message_output = a->second;
            if (this->function->message_output_optional) {
                a->second->optional_outputs++;
            }
        } else {
            THROW exception::InvalidMessageType("Message ('%s') type '%s' does not match type '%s' applied to FLAMEGPU_AGENT_FUNCTION ('%s'), "
                "in AgentFunctionDescription::setMessageOutput().",
                message_name.c_str(), demangledClassName.c_str(), msg_out_classname.c_str(), this->function->name.c_str());
        }
    } else {
        THROW exception::InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageOutput().",
            mdl->name.c_str(), message_name.c_str());
    }
}
void AgentFunctionDescription::setMessageOutput(MsgBruteForce::Description &message) {
    if (message.model.lock() != function->description->model.lock()) {
        THROW exception::DifferentModel("Attempted to use agent description from a different model, "
            "in AgentFunctionDescription::setAgentOutput().");
    }
    if (auto other = function->message_input.lock()) {
        if (message.getName() == other->name) {
            THROW exception::InvalidMessageName("Message '%s' is already bound as message input in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageOutput().",
                message.getName().c_str(), function->name.c_str());
        }
    }
    // Clear old value
    if (this->function->message_output_optional) {
        if (auto b = this->function->message_output.lock()) {
            b->optional_outputs--;
        }
    }
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->messages.find(message.getName());
    if (a != mdl->messages.end()) {
        if (a->second->description.get() == &message) {
            // Just compare the classname is the same, to allow for the various approaches to namespace use. This should only be required for RTC functions.
            auto msg_out_classname = util::detail::cxxname::getUnqualifiedName(this->function->msg_out_type);
            auto demangledClassName = util::detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(a->second->getType()));
            if (msg_out_classname == demangledClassName) {
                this->function->message_output = a->second;
                if (this->function->message_output_optional) {
                    a->second->optional_outputs++;
                }
            } else {
                THROW exception::InvalidMessageType("Message ('%s') type '%s' does not match type '%s' applied to FLAMEGPU_AGENT_FUNCTION ('%s'), "
                    "in AgentFunctionDescription::setMessageOutput().",
                    a->second->name.c_str(), demangledClassName.c_str(), msg_out_classname.c_str(), this->function->name.c_str());
            }
        } else {
            THROW exception::InvalidMessage("Message '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setMessageOutput().",
                message.getName().c_str(), mdl->name.c_str());
        }
    } else {
        THROW exception::InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageOutput()\n",
            mdl->name.c_str(), message.getName().c_str());
    }
}
void AgentFunctionDescription::setMessageOutputOptional(const bool &output_is_optional) {
    if (output_is_optional != this->function->message_output_optional) {
        this->function->message_output_optional = output_is_optional;
        if (auto b = this->function->message_output.lock()) {
            if (output_is_optional)
                b->optional_outputs++;
            else
                b->optional_outputs--;
        }
    }
}
void AgentFunctionDescription::setAgentOutput(const std::string &agent_name, const std::string state) {
    // Set new
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->agents.find(agent_name);
    if (a != mdl->agents.end()) {
        // Check agent state is valid
        if (a->second->states.find(state)!= a->second->states.end()) {    // Clear old value
            if (auto b = this->function->agent_output.lock()) {
                b->agent_outputs--;
            }
            this->function->agent_output = a->second;
            this->function->agent_output_state = state;
            a->second->agent_outputs++;  // Mark inside agent that we are using it as an output
        } else {
            THROW exception::InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setAgentOutput().",
                agent_name.c_str(), state.c_str());
        }
    } else {
        THROW exception::InvalidAgentName("Model ('%s') does not contain agent '%s', "
            "in AgentFunctionDescription::setAgentOutput().",
            mdl->name.c_str(), agent_name.c_str());
    }
}
void AgentFunctionDescription::setAgentOutput(AgentDescription &agent, const std::string state) {
    if (agent.model.lock() != function->description->model.lock()) {
        THROW exception::DifferentModel("Attempted to use agent description from a different model, "
            "in AgentFunctionDescription::setAgentOutput().");
    }
    // Set new
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->agents.find(agent.getName());
    if (a != mdl->agents.end()) {
        if (a->second->description.get() == &agent) {
            // Check agent state is valid
            if (a->second->states.find(state) != a->second->states.end()) {
                // Clear old value
                if (auto b = this->function->agent_output.lock()) {
                    b->agent_outputs--;
                }
                this->function->agent_output = a->second;
                this->function->agent_output_state = state;
                a->second->agent_outputs++;  // Mark inside agent that we are using it as an output
            } else {
                THROW exception::InvalidStateName("Agent ('%s') does not contain state '%s', "
                    "in AgentFunctionDescription::setAgentOutput().",
                    agent.getName().c_str(), state.c_str());
            }
        } else {
            THROW exception::InvalidMessage("Agent '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setAgentOutput().",
                agent.getName().c_str(), mdl->name.c_str());
        }
    } else {
        THROW exception::InvalidMessageName("Model ('%s') does not contain agent '%s', "
            "in AgentFunctionDescription::setAgentOutput().",
            mdl->name.c_str(), agent.getName().c_str());
    }
}
void AgentFunctionDescription::setAllowAgentDeath(const bool &has_death) {
    function->has_agent_death = has_death;
}

void AgentFunctionDescription::setRTCFunctionCondition(std::string func_cond_src) {
    // Use Regex to get agent function name
    std::regex rgx(R"###(.*FLAMEGPU_AGENT_FUNCTION_CONDITION\([ \t]*(\w+)[ \t]*)###");
    std::smatch match;
    std::string func_cond_name;
    if (std::regex_search(func_cond_src, match, rgx)) {
        if (match.size() == 2) {
            func_cond_name = match[1];
            // set the runtime agent function condition source in agent function data
            function->rtc_func_condition_name = func_cond_name;
            function->rtc_condition_source = func_cond_src;
            // TODO: Does this need emplacing in CUDAAgent?
        } else {
            THROW exception::InvalidAgentFunc("Runtime agent function condition is missing FLAMEGPU_AGENT_FUNCTION_CONDITION arguments e.g. 'FLAMEGPU_AGENT_FUNCTION_CONDITION(func_name)', "
                "in AgentDescription::setRTCFunctionCondition().");
        }
    } else {
        THROW exception::InvalidAgentFunc("Runtime agent function('%s') is missing FLAMEGPU_AGENT_FUNCTION_CONDITION, "
            "in AgentDescription::setRTCFunctionCondition().");
    }

    // append jitify program string and include
    std::string func_cond_src_str = std::string(func_cond_name + "_program\n").append("#include \"flamegpu/runtime/DeviceAPI.h\"\n").append(func_cond_src);

    // update the agent function data
    function->rtc_func_condition_name = func_cond_name;
    function->rtc_condition_source = func_cond_src_str;
}

MsgBruteForce::Description &AgentFunctionDescription::MessageInput() {
    if (auto m = function->message_input.lock())
        return *m->description;
    THROW exception::OutOfBoundsException("Message input has not been set, "
        "in AgentFunctionDescription::MessageInput().");
}
MsgBruteForce::Description &AgentFunctionDescription::MessageOutput() {
    if (auto m = function->message_output.lock())
        return *m->description;
    THROW exception::OutOfBoundsException("Message output has not been set, "
        "in AgentFunctionDescription::MessageOutput().");
}
bool &AgentFunctionDescription::MessageOutputOptional() {
    return function->message_output_optional;
}
bool &AgentFunctionDescription::AllowAgentDeath() {
    return function->has_agent_death;
}

/**
 * Const Accessors
 */
std::string AgentFunctionDescription::getName() const {
    return function->name;
}
std::string AgentFunctionDescription::getInitialState() const {
    return function->initial_state;
}
std::string AgentFunctionDescription::getEndState() const {
    return function->end_state;
}
const MsgBruteForce::Description &AgentFunctionDescription::getMessageInput() const {
    if (auto m = function->message_input.lock())
        return *m->description;
    THROW exception::OutOfBoundsException("Message input has not been set, "
        "in AgentFunctionDescription::getMessageInput().");
}
const MsgBruteForce::Description &AgentFunctionDescription::getMessageOutput() const {
    if (auto m = function->message_output.lock())
        return *m->description;
    THROW exception::OutOfBoundsException("Message output has not been set, "
        "in AgentFunctionDescription::getMessageOutput().");
}
bool AgentFunctionDescription::getMessageOutputOptional() const {
    return this->function->message_output_optional;
}
const AgentDescription &AgentFunctionDescription::getAgentOutput() const {
    if (auto a = function->agent_output.lock())
        return *a->description;
    THROW exception::OutOfBoundsException("Agent output has not been set, "
        "in AgentFunctionDescription::getAgentOutput().");
}
std::string AgentFunctionDescription::getAgentOutputState() const {
    if (auto a = function->agent_output.lock())
        return function->agent_output_state;
    THROW exception::OutOfBoundsException("Agent output has not been set, "
        "in AgentFunctionDescription::getAgentOutputState().");
}
bool AgentFunctionDescription::getAllowAgentDeath() const {
    return function->has_agent_death;
}

bool AgentFunctionDescription::hasMessageInput() const {
    return function->message_input.lock() != nullptr;
}
bool AgentFunctionDescription::hasMessageOutput() const {
    return function->message_output.lock() != nullptr;
}
bool AgentFunctionDescription::hasAgentOutput() const {
    return function->agent_output.lock() != nullptr;
}
bool AgentFunctionDescription::hasFunctionCondition() const {
    return function->condition != nullptr;
}
AgentFunctionWrapper *AgentFunctionDescription::getFunctionPtr() const {
    return function->func;
}
AgentFunctionConditionWrapper *AgentFunctionDescription::getConditionPtr() const {
    return function->condition;
}

bool AgentFunctionDescription::isRTC() const {
    return !function->rtc_source.empty();
}

AgentFunctionDescription& AgentDescription::newRTCFunction(const std::string& function_name, const std::string& func_src) {
    if (agent->functions.find(function_name) == agent->functions.end()) {
        // append jitify program string and include
        std::string func_src_str = std::string(function_name + "_program\n").append("#include \"flamegpu/runtime/DeviceAPI.h\"\n").append(func_src);
        // Use Regex to get agent function name, and input/output message type
        std::regex rgx(R"###(.*FLAMEGPU_AGENT_FUNCTION\([ \t]*(\w+),[ \t]*([:\w]+),[ \t]*([:\w]+)[ \t]*\))###");
        std::smatch match;
        if (std::regex_search(func_src_str, match, rgx)) {
            if (match.size() == 4) {
                std::string code_func_name = match[1];  // not yet clear if this is required
                std::string in_type_name = match[2];
                std::string out_type_name = match[3];
                // set the runtime agent function source in agent function data
                auto rtn = std::shared_ptr<AgentFunctionData>(new AgentFunctionData(this->agent->shared_from_this(), function_name, func_src_str, in_type_name, out_type_name, code_func_name));
                agent->functions.emplace(function_name, rtn);
                return *rtn->description;
            } else {
                THROW exception::InvalidAgentFunc("Runtime agent function('%s') is missing FLAMEGPU_AGENT_FUNCTION arguments e.g. (func_name, message_input_type, message_output_type), "
                    "in AgentDescription::newRTCFunction().",
                    agent->name.c_str());
            }
        } else {
            THROW exception::InvalidAgentFunc("Runtime agent function('%s') is missing FLAMEGPU_AGENT_FUNCTION, "
                "in AgentDescription::newRTCFunction().",
                agent->name.c_str());
        }
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') already contains function '%s', "
        "in AgentDescription::newRTCFunction().",
        agent->name.c_str(), function_name.c_str());
}

AgentFunctionDescription& AgentDescription::newRTCFunctionFile(const std::string& function_name, const std::string& file_path) {
    if (agent->functions.find(function_name) == agent->functions.end()) {
        // Load file and forward to regular RTC method
        std::ifstream file;
        file.open(file_path);
        if (file.is_open()) {
            std::stringstream sstream;
            sstream << file.rdbuf();
            const std::string func_src = sstream.str();
            return newRTCFunction(function_name, func_src);
        }
        THROW exception::InvalidFilePath("Unable able to open file '%s', "
            "in AgentDescription::newRTCFunctionFile().",
            file_path.c_str());
    }
    THROW exception::InvalidAgentFunc("Agent ('%s') already contains function '%s', "
        "in AgentDescription::newRTCFunctionFile().",
        agent->name.c_str(), function_name.c_str());
}

}  // namespace flamegpu
