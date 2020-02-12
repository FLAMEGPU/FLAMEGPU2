#include "flamegpu/runtime/messaging/Spatial3D.h"

void MsgSpatial3D::Out::setLocation(const float &x, const float &y, const float &z) const {
    unsigned int index = (blockDim.x * blockIdx.x) + threadIdx.x;  // + d_message_count;

    // set the variables using curve
    Curve::setVariable<float>("x", combined_hash, x, index);
    Curve::setVariable<float>("y", combined_hash, y, index);
    Curve::setVariable<float>("z", combined_hash, z, index);

    // Set scan flag incase the message is optional
    flamegpu_internal::CUDAScanCompaction::ds_message_configs[streamId].scan_flag[index] = 1;
}
namespace {
    __host__ __device__ __forceinline__ GridPos3D getGridPosition_2(const MsgSpatial3D::MetaData *md, float x, float y, float z)
    {
        //Clamp each grid coord to 0<=x<dim
        unsigned int gridPos[3] = {
            (unsigned int)floor((x / md->environmentWidth[0])*md->gridDim[0]),
            (unsigned int)floor((y / md->environmentWidth[0])*md->gridDim[0]),
            (unsigned int)floor((z / md->environmentWidth[0])*md->gridDim[0])
        };
        GridPos3D rtn = {
            gridPos[0] > md->gridDim[0] - 1 ? md->gridDim[0] - 1 : gridPos[0],
            gridPos[1] > md->gridDim[1] - 1 ? md->gridDim[1] - 1 : gridPos[1],
            gridPos[2] > md->gridDim[2] - 1 ? md->gridDim[2] - 1 : gridPos[2]
        };
        return rtn;
    }
    __host__ __device__ __forceinline__ unsigned int getHash3D_2(const MsgSpatial3D::MetaData *md, const GridPos3D &xyz)
    {
        //Bound gridPos to gridDimensions
        unsigned int gridPos[3] = {
            xyz.x > md->gridDim[0] - 1 ? md->gridDim[0] - 1 : xyz.x,
            xyz.y > md->gridDim[1] - 1 ? md->gridDim[1] - 1 : xyz.y,
            xyz.z > md->gridDim[2] - 1 ? md->gridDim[2] - 1 : xyz.z
        };
        //Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
        return (unsigned int)(
            (gridPos[2] * md->gridDim[0] * md->gridDim[1]) + //z
            (gridPos[1] * md->gridDim[0]) +					 //y
            gridPos[0]);                                     //x
    }
}
__device__ MsgSpatial3D::In::Filter::Filter(const MetaData* _metadata, const Curve::NamespaceHash &_combined_hash, const float& x, const float& y, const float& z)
    : metadata(_metadata)
    , combined_hash(_combined_hash) {
    loc[0] = x;
    loc[1] = y;
    loc[2] = z;
    cell = getGridPosition_2(_metadata, x, y, z);
}
MsgSpatial3D::In::Filter::Message& MsgSpatial3D::In::Filter::Message::operator++()
{
    cell_index++;
    bool move_strip = cell_index >= cell_index_max;
    while (move_strip) {
        nextStrip();
        if (relative_cell[0] < 2) {
            // Calculate the strips start and end hash
            unsigned int start_hash = getHash3D_2(_parent.metadata, { _parent.cell.x - 1, _parent.cell.y + relative_cell[0], _parent.cell.x + relative_cell[1] });
            unsigned int end_hash = getHash3D_2(_parent.metadata, { _parent.cell.x + 1, _parent.cell.y + relative_cell[0], _parent.cell.x + relative_cell[1] });
            // Lookup start and end indicies from PBM
            cell_index = _parent.metadata->PBM[start_hash];
            cell_index_max = _parent.metadata->PBM[end_hash + 1];
        }
        else
        {
            cell_index = 0;
            cell_index_max = 1;
        }
        move_strip = cell_index >= cell_index_max;
    }
    return *this;
}
