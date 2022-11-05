#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceDevice.cuh"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"

namespace flamegpu {

void MessageBruteForce::CUDAModelHandler::init(CUDAScatter &, unsigned int, cudaStream_t stream) {
    allocateMetaDataDevicePtr(stream);
    // Allocate messages
    hd_metadata.length = 0;  // This value should already be 0
    gpuErrchk(cudaMemcpyAsync(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));  // This could probably be skipped/delayed safely
}

void MessageBruteForce::CUDAModelHandler::allocateMetaDataDevicePtr(cudaStream_t stream) {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
    }
}

void MessageBruteForce::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_metadata != nullptr) {
        gpuErrchk(cudaFree(d_metadata));
    }
    d_metadata = nullptr;
}

void MessageBruteForce::CUDAModelHandler::buildIndex(CUDAScatter &, unsigned int, cudaStream_t stream) {
    unsigned int newLength = this->sim_message.getMessageCount();
    if (newLength != hd_metadata.length) {
        hd_metadata.length = newLength;
        gpuErrchk(cudaMemcpyAsync(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice, stream));  // Not Pinned
        gpuErrchk(cudaStreamSynchronize(stream));  // This could probably be skipped/delayed safely if in the right stream
    }
}

MessageBruteForce::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : description(new Description(model, this))
    , name(message_name)
    , optional_outputs(0) { }
MessageBruteForce::Data::~Data() {}
MessageBruteForce::Data::Data(const std::shared_ptr<const ModelData> &model, const MessageBruteForce::Data &other)
    : variables(other.variables)
    , description(model ? new Description(model, this) : nullptr)
    , name(other.name)
    , optional_outputs(other.optional_outputs) { }
MessageBruteForce::Data *MessageBruteForce::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new MessageBruteForce::Data(newParent, *this);
}
bool MessageBruteForce::Data::operator==(const MessageBruteForce::Data& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name
        && variables.size() == rhs.variables.size()) {
            {  // Compare variables
                for (auto &v : variables) {
                    auto _v = rhs.variables.find(v.first);
                    if (_v == rhs.variables.end())
                        return false;
                    if (v.second.type_size != _v->second.type_size
                        || v.second.type != _v->second.type
                        || v.second.elements != _v->second.elements)
                        return false;
                }
            }
            return true;
    }
    return false;
}
bool MessageBruteForce::Data::operator!=(const MessageBruteForce::Data& rhs) const {
    return !operator==(rhs);
}

std::unique_ptr<MessageSpecialisationHandler> MessageBruteForce::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MessageSpecialisationHandler>(new MessageBruteForce::CUDAModelHandler(owner));
}

flamegpu::MessageSortingType flamegpu::MessageBruteForce::Data::getSortingType() const {
    return flamegpu::MessageSortingType::none;
}

// Used for the MessageBruteForce::Data::getType() type and derived methods
std::type_index MessageBruteForce::Data::getType() const { return std::type_index(typeid(MessageBruteForce)); }


/**
* Constructors
*/
MessageBruteForce::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const description)
    : model(_model)
    , message(description) { }

bool MessageBruteForce::Description::operator==(const MessageBruteForce::Description& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageBruteForce::Description::operator!=(const MessageBruteForce::Description& rhs) const {
    return !(*this == rhs);
}

/**
* Const Accessors
*/
std::string MessageBruteForce::Description::getName() const {
    return message->name;
}

const std::type_index& MessageBruteForce::Description::getVariableType(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type;
    }
    THROW exception::InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableType().",
        message->name.c_str(), variable_name.c_str());
}
size_t MessageBruteForce::Description::getVariableSize(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type_size;
    }
    THROW exception::InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableSize().",
        message->name.c_str(), variable_name.c_str());
}
flamegpu::size_type MessageBruteForce::Description::getVariableLength(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.elements;
    }
    THROW exception::InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageBruteForce::getVariableLength().",
        message->name.c_str(), variable_name.c_str());
}
flamegpu::size_type MessageBruteForce::Description::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX variables
    return static_cast<flamegpu::size_type>(message->variables.size());
}
bool MessageBruteForce::Description::hasVariable(const std::string &variable_name) const {
    return message->variables.find(variable_name) != message->variables.end();
}

}  // namespace flamegpu
