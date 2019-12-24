#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign

/**
 * Constructors
 */
MessageDescription::MessageDescription(ModelData *const _model, MessageData *const description)
    : model(_model)
    , message(description) { }

bool MessageDescription::operator==(const MessageDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageDescription::operator!=(const MessageDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
std::string MessageDescription::getName() const {
    return message->name;
}

std::type_index MessageDescription::getVariableType(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableType().",
        message->name.c_str(), variable_name.c_str());
}
size_t MessageDescription::getVariableSize(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type_size;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableSize().",
        message->name.c_str(), variable_name.c_str());
}
ModelData::size_type MessageDescription::getVariableLength(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.elements;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableLength().",
        message->name.c_str(), variable_name.c_str());
}
ModelData::size_type MessageDescription::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX variables
    return static_cast<ModelData::size_type>(message->variables.size());
}
bool MessageDescription::hasVariable(const std::string &variable_name) const {
    return message->variables.find(variable_name) != message->variables.end();
}


Spatial2DMessageDescription::Spatial2DMessageDescription(ModelData *const _model, Spatial2DMessageData *const data)
    : MessageDescription(_model, data) { }

void Spatial2DMessageDescription::setRadius(const float &r) {
    reinterpret_cast<Spatial2DMessageData *>(message)->radius = r;
}
void Spatial2DMessageDescription::setMinX(const float &x) {
    reinterpret_cast<Spatial2DMessageData *>(message)->minX = x;
}
void Spatial2DMessageDescription::setMinY(const float &y) {
    reinterpret_cast<Spatial2DMessageData *>(message)->minY = y;
}
void Spatial2DMessageDescription::setMaxX(const float &x) {
    reinterpret_cast<Spatial2DMessageData *>(message)->maxX = x;
}
void Spatial2DMessageDescription::setMaxY(const float &y) {
    reinterpret_cast<Spatial2DMessageData *>(message)->maxY = y;
}

float &Spatial2DMessageDescription::Radius() {
    return reinterpret_cast<Spatial2DMessageData *>(message)->radius;
}
float &Spatial2DMessageDescription::MinX() {
    return reinterpret_cast<Spatial2DMessageData *>(message)->minX;
}
float &Spatial2DMessageDescription::MinY() {
    return reinterpret_cast<Spatial2DMessageData *>(message)->minY;
}
float &Spatial2DMessageDescription::MaxX() {
    return reinterpret_cast<Spatial2DMessageData *>(message)->maxX;
}
float &Spatial2DMessageDescription::MaxY() {
    return reinterpret_cast<Spatial2DMessageData *>(message)->maxY;
}

float Spatial2DMessageDescription::getRadius() const {
    return reinterpret_cast<Spatial2DMessageData *>(message)->radius;
}
float Spatial2DMessageDescription::getMinX() const {
    return reinterpret_cast<Spatial2DMessageData *>(message)->minX;
}
float Spatial2DMessageDescription::getMinY() const {
    return reinterpret_cast<Spatial2DMessageData *>(message)->minY;
}
float Spatial2DMessageDescription::getMaxX() const {
    return reinterpret_cast<Spatial2DMessageData *>(message)->maxX;
}
float Spatial2DMessageDescription::getMaxY() const {
    return reinterpret_cast<Spatial2DMessageData *>(message)->maxY;
}


Spatial3DMessageDescription::Spatial3DMessageDescription(ModelData *const _model, Spatial3DMessageData *const data)
    : Spatial2DMessageDescription(_model, data) { }

void Spatial3DMessageDescription::setMinZ(const float &z) {
    reinterpret_cast<Spatial3DMessageData *>(message)->minZ = z;
}
void Spatial3DMessageDescription::setMaxZ(const float &z) {
    reinterpret_cast<Spatial3DMessageData *>(message)->maxZ = z;
}
float &Spatial3DMessageDescription::MinZ() {
    return reinterpret_cast<Spatial3DMessageData *>(message)->minZ;
}
float &Spatial3DMessageDescription::MaxZ() {
    return reinterpret_cast<Spatial3DMessageData *>(message)->maxZ;
}
float Spatial3DMessageDescription::getMinZ() const {
    return reinterpret_cast<Spatial3DMessageData *>(message)->minZ;
}
float Spatial3DMessageDescription::getMaxZ() const {
    return reinterpret_cast<Spatial3DMessageData *>(message)->maxZ;
}
