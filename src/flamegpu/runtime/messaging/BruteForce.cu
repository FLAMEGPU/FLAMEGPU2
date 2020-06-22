#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign
#include "flamegpu/gpu/CUDAMessage.h"

void MsgBruteForce::CUDAModelHandler::allocateMetaDataDevicePtr() {
    if (d_metadata == nullptr) {
        gpuErrchk(cudaMalloc(&d_metadata, sizeof(MetaData)));
    }
}

void MsgBruteForce::CUDAModelHandler::freeMetaDataDevicePtr() {
    if (d_metadata != nullptr) {
        gpuErrchk(cudaFree(d_metadata));
    }
    d_metadata = nullptr;
}

void MsgBruteForce::CUDAModelHandler::buildIndex(CUDAScatter &, const unsigned int &) {
    unsigned int newLength = this->sim_message.getMessageCount();
    if (newLength != hd_metadata.length) {
        hd_metadata.length = newLength;
        gpuErrchk(cudaMemcpy(d_metadata, &hd_metadata, sizeof(MetaData), cudaMemcpyHostToDevice));
    }
}

MsgBruteForce::Data::Data(const std::shared_ptr<const ModelData> &model, const std::string &message_name)
    : description(new Description(model, this))
    , name(message_name)
    , optional_outputs(0) { }
MsgBruteForce::Data::~Data() {}
MsgBruteForce::Data::Data(const std::shared_ptr<const ModelData> &model, const MsgBruteForce::Data &other)
    : variables(other.variables)
    , description(model ? new Description(model, this) : nullptr)
    , name(other.name)
    , optional_outputs(other.optional_outputs) { }
MsgBruteForce::Data *MsgBruteForce::Data::clone(const std::shared_ptr<const ModelData> &newParent) {
    return new MsgBruteForce::Data(newParent, *this);
}
bool MsgBruteForce::Data::operator==(const MsgBruteForce::Data& rhs) const {
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
bool MsgBruteForce::Data::operator!=(const MsgBruteForce::Data& rhs) const {
    return !operator==(rhs);
}

std::unique_ptr<MsgSpecialisationHandler> MsgBruteForce::Data::getSpecialisationHander(CUDAMessage &owner) const {
    return std::unique_ptr<MsgSpecialisationHandler>(new MsgBruteForce::CUDAModelHandler(owner));
}

// Used for the MsgBruteForce::Data::getType() type and derived methods
std::type_index MsgBruteForce::Data::getType() const { return std::type_index(typeid(MsgBruteForce)); }


/**
* Constructors
*/
MsgBruteForce::Description::Description(const std::shared_ptr<const ModelData> &_model, Data *const description)
    : model(_model)
    , message(description) { }

bool MsgBruteForce::Description::operator==(const MsgBruteForce::Description& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MsgBruteForce::Description::operator!=(const MsgBruteForce::Description& rhs) const {
    return !(*this == rhs);
}

/**
* Const Accessors
*/
std::string MsgBruteForce::Description::getName() const {
    return message->name;
}

std::type_index MsgBruteForce::Description::getVariableType(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableType().",
        message->name.c_str(), variable_name.c_str());
}
size_t MsgBruteForce::Description::getVariableSize(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type_size;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableSize().",
        message->name.c_str(), variable_name.c_str());
}
ModelData::size_type MsgBruteForce::Description::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX variables
    return static_cast<ModelData::size_type>(message->variables.size());
}
bool MsgBruteForce::Description::hasVariable(const std::string &variable_name) const {
    return message->variables.find(variable_name) != message->variables.end();
}
