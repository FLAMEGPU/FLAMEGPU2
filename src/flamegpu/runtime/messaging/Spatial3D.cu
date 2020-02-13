#include "flamegpu/runtime/messaging/Spatial3D.h"

__device__ void MsgSpatial3D::Out::setLocation(const float &x, const float &y, const float &z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    Curve::setVariable<float>("x", combined_hash, x, index);
    Curve::setVariable<float>("y", combined_hash, y, index);
    Curve::setVariable<float>("z", combined_hash, z, index);

    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].scan_flag[index] = 1;
}

__device__ MsgSpatial3D::In::Filter::Filter(const MetaData* _metadata, const Curve::NamespaceHash &_combined_hash, const float& x, const float& y, const float& z)
    : metadata(_metadata)
    , combined_hash(_combined_hash) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
    cell = Spatial3D::getGridPosition(_metadata, x, y, z);
}
__device__ MsgSpatial3D::In::Filter::Message& MsgSpatial3D::In::Filter::Message::operator++() {
    cell_index++;
    bool move_strip = cell_index >= cell_index_max;
    while (move_strip) {
        nextStrip();
        if (relative_cell[0] < 2) {
            // Calculate the strips start and end hash
            unsigned int start_hash = Spatial3D::getHash(_parent.metadata, { _parent.cell.x - 1, _parent.cell.y + relative_cell[0], _parent.cell.z + relative_cell[1] });
            unsigned int end_hash = Spatial3D::getHash(_parent.metadata, { _parent.cell.x + 1, _parent.cell.y + relative_cell[0], _parent.cell.z + relative_cell[1] });
            // Lookup start and end indicies from PBM
            cell_index = _parent.metadata->PBM[start_hash];
            cell_index_max = _parent.metadata->PBM[end_hash + 1];
        } else {
            cell_index = 0;
            cell_index_max = 1;
        }
        move_strip = cell_index >= cell_index_max;
    }
    return *this;
}
