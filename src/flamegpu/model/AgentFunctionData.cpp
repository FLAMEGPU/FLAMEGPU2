#include "flamegpu/model/AgentFunctionData.cuh"

#include <string>
#include <memory>

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"
#include "flamegpu/detail/cxxname.hpp"

namespace flamegpu {

AgentFunctionData::AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string &function_name, AgentFunctionWrapper *agent_function, const std::string &in_type, const std::string &out_type)
    : model(_parent->model)
    , func(agent_function)
    , initial_state(_parent->initial_state)
    , end_state(_parent->initial_state)
    , message_output_optional(false)
    , has_agent_death(false)
    , condition(nullptr)
    , parent(_parent)
    , name(function_name)
    , message_in_type(in_type)
    , message_out_type(out_type) { }
AgentFunctionData::AgentFunctionData(std::shared_ptr<AgentData> _parent, const std::string& function_name, const std::string &rtc_function_src, const std::string &in_type, const std::string& out_type, const std::string& code_func_name)
    : model(_parent->model)
    , func(nullptr)
    , rtc_source(rtc_function_src)
    , rtc_func_name(code_func_name)
    , initial_state(_parent->initial_state)
    , end_state(_parent->initial_state)
    , message_output_optional(false)
    , has_agent_death(false)
    , condition(nullptr)
    , parent(_parent)
    , name(function_name)
    , message_in_type(in_type)
    , message_out_type(out_type) { }

AgentFunctionData::AgentFunctionData(const std::shared_ptr<const ModelData> &_model, std::shared_ptr<AgentData> _parent, const AgentFunctionData &other)
    : model(_model)
    , func(other.func)
    , rtc_source(other.rtc_source)
    , rtc_func_name(other.rtc_func_name)
    , initial_state(other.initial_state)
    , end_state(other.end_state)
    , message_output_optional(other.message_output_optional)
    , agent_output_state(other.agent_output_state)
    , has_agent_death(other.has_agent_death)
    , condition(other.condition)
    , rtc_condition_source(other.rtc_condition_source)
    , rtc_func_condition_name(other.rtc_func_condition_name)
    , parent(_parent)
    , name(other.name)
    , message_in_type(other.message_in_type)
    , message_out_type(other.message_out_type) {
    // Manually perform lookup copies
    if (_model) {
        if (auto a = other.message_input.lock()) {
            auto _m = _model->messages.find(a->name);
            if (_m != _model->messages.end()) {
                message_input = _m->second;
            }
        } else if (detail::cxxname::getUnqualifiedName(other.message_in_type) != detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(std::type_index(typeid(MessageNone))))) {
            THROW exception::InvalidMessageType(
                "Function '%s' is missing bound input message of type '%s', type provided was '%s'.", other.name.c_str(),
                detail::cxxname::getUnqualifiedName(other.message_in_type).c_str(),
                detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(std::type_index(typeid(MessageNone)))).c_str());
        }
        if (auto a = other.message_output.lock()) {
            auto _m = _model->messages.find(a->name);
            if (_m != _model->messages.end()) {
                message_output = _m->second;
            }
        } else if (detail::cxxname::getUnqualifiedName(other.message_out_type) != detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(std::type_index(typeid(MessageNone))))) {
            THROW exception::InvalidMessageType(
                "Function '%s' is missing bound output message of type '%s', type provided was '%s'.", other.name.c_str(),
                detail::cxxname::getUnqualifiedName(other.message_out_type).c_str(),
                detail::cxxname::getUnqualifiedName(detail::curve::CurveRTCHost::demangle(std::type_index(typeid(MessageNone)))).c_str());
        }
        if (auto a = other.agent_output.lock()) {
            auto _a = _model->agents.find(a->name);
            if (_a != _model->agents.end()) {
                agent_output = _a->second;
            }
        }
    }
}

bool AgentFunctionData::operator==(const AgentFunctionData &rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if ((name == rhs.name)
        //  && (model.lock() == rhs.model.lock())  // Don't check weak pointers
        && (func == rhs.func)
        && (rtc_source == rhs.rtc_source)
        && (rtc_func_name == rhs.rtc_func_name)
        && (initial_state == rhs.initial_state)
        && (end_state == rhs.end_state)
        && (message_output_optional == rhs.message_output_optional)
        && (agent_output_state == rhs.agent_output_state)
        && (has_agent_death == rhs.has_agent_death)
        && (condition == rhs.condition)
        && (rtc_condition_source == rhs.rtc_condition_source)
        && (rtc_func_condition_name == rhs.rtc_func_condition_name)) {
        // Test weak pointers
        {   // parent
            auto a = parent.lock();
            auto b = rhs.parent.lock();
            if (a && b) {
                // We can't call equality here, as that would be infinite recursion
                if (a->name != b->name ||
                    a->functions.size() != b->functions.size()) {
                    return false;
                }
            } else {
                return false;
            }
        }
        {  // message_input
            auto a = message_input.lock();
            auto b = rhs.message_input.lock();
            if (a && b) {
                if (*a != *b)
                    return false;
            } else if ((a && !b) || (!a && b)) {
                return false;
            }
        }
        {  // message_output
            auto a = message_output.lock();
            auto b = rhs.message_output.lock();
            if (a && b) {
                if (*a != *b)
                    return false;
            } else if ((a && !b) || (!a && b)) {
                return false;
            }
        }
        {  // agent_output
            auto a = agent_output.lock();
            auto b = rhs.agent_output.lock();
            if (a && b) {  // Comparing agents here is unsafe, as agent might link back to this same function, so do a reduced comparison
                if (a->name != b->name
                || a->initial_state != b->initial_state
                || a->agent_outputs != b->agent_outputs
                || a->keepDefaultState != b->keepDefaultState
                || a->states.size() != b->states.size()
                || a->functions.size() != b->functions.size()
                || a->variables.size() != b->variables.size()
                || a->states != b->states)
                    return false;
                {  // Compare agent variables
                    for (auto &v : a->variables) {
                        auto _v = b->variables.find(v.first);
                        if (_v == b->variables.end())
                            return false;
                        if (v.second.type_size != _v->second.type_size || v.second.type != _v->second.type || v.second.elements != _v->second.elements)
                            return false;
                    }
                }
                {  // Compare agent functions (compare name only)
                    for (auto &v : a->functions) {
                        auto _v = b->functions.find(v.first);
                        if (_v == b->functions.end())
                            return false;
                    }
                }
            } else if ((a && !b) || (!a && b)) {
                return false;
            }
        }
        return true;
    }
    return false;
}
bool AgentFunctionData::operator==(const CAgentFunctionDescription& rhs) const {
    return *this == *rhs.function;
}
bool AgentFunctionData::operator!=(const AgentFunctionData &rhs) const {
    return !operator==(rhs);
}

}  // namespace flamegpu
