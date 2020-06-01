#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3DHOST_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3DHOST_H_

#include <memory>
#include <string>

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"
#include "flamegpu/runtime/messaging/Spatial2D/Spatial2DHost.h"
#include "flamegpu/runtime/messaging/BruteForce/BruteForceHost.h"

/**
 * CUDA host side handler of spatial messages
 * Allocates memory for and constructs PBM
 */
class MsgSpatial3D::CUDAModelHandler : public MsgSpecialisationHandler {
 public:
	/**
	 * Constructor
	 * 
	 * Initialises metadata, decides PBM size etc
	 * 
	 * @param a Parent CUDAMessage, used to access message settings, data ptrs etc
	 */
     explicit CUDAModelHandler(CUDAMessage& a);
	/**
	 * Destructor
	 * Frees all alocated memory
	 */
	~CUDAModelHandler() override { }
	/**
	 * Reconstructs the partition boundary matrix
	 * This should be called before reading newly output messages
	 */
	void buildIndex() override;
	/**
	 * Allocates memory for the constructed index.
	 * The memory allocation is checked by build index.
	 */
	void allocateMetaDataDevicePtr() override;
	/**
	 * Releases memory for the constructed index.
	 */
	void freeMetaDataDevicePtr() override;
	/**
	 * Returns a pointer to the metadata struct, this is required for reading the message data
	 */
	const void *getMetaDataDevicePtr() const override { return d_data; }

 private:
	/**
	 * Resizes the cub temp memory
	 * Currently assumed that bounds of environment/rad never change
	 * So this is only called from the constructor.
	 * If it were called elsewhere, it would need to be changed to resize d_histogram too
	 */
	void resizeCubTemp();
	/**
	 * Resizes the key value store, this scales with agent count
	 * @param newSize The new number of agents to represent
	 * @note This only scales upwards, it will never reduce the size
	 */
	void resizeKeysVals(const unsigned int &newSize);
	/**
	 * Number of bins, arrays are +1 this length
	 */
	unsigned int binCount = 0;
	/**
	 * Size of currently allocated temp storage memory for cub
	 */
	size_t d_CUB_temp_storage_bytes = 0;
	/**
	 * Pointer to currently allocated temp storage memory for cub
	 */
	unsigned int *d_CUB_temp_storage = nullptr;
	/**
	 * Pointer to array used for histogram
	 */
	unsigned int *d_histogram = nullptr;
	/**
	 * Arrays used to store indices when sorting messages
	 */
	unsigned int *d_keys = nullptr, *d_vals = nullptr;
	/**
	 * Size currently allocated to d_keys, d_vals arrays
	 */
	size_t d_keys_vals_storage_bytes = 0;
	/**
	 * Host copy of metadata struct
	 */
	MetaData hd_data;
	/**
	 * Pointer to device copy of metadata struct
	 */
	MetaData *d_data = nullptr;
	/**
	 * Owning CUDAMessage, provides access to message storage etc
	 */
	CUDAMessage &sim_message;
};

/**
 * Internal data representation of Spatial3D messages within model description hierarchy
 * @see Description
 */
struct MsgSpatial3D::Data : public MsgSpatial2D::Data {
	friend class ModelDescription;
	friend struct ModelData;
	float minZ;
	float maxZ;
	virtual ~Data() = default;

	std::unique_ptr<MsgSpecialisationHandler> getSpecialisationHander(CUDAMessage &owner) const override;

	/**
	* Used internally to validate that the corresponding Msg type is attached via the agent function shim.
	* @return The std::type_index of the Msg type which must be used.
	*/
	std::type_index getType() const override;

 protected:
	Data *clone(ModelData *const newParent) override;
	/**
	* Copy constructor
	* This is unsafe, should only be used internally, use clone() instead
	*/
	Data(ModelData *const, const Data &other);
	/**
	* Normal constructor, only to be called by ModelDescription
	*/
	Data(ModelData *const, const std::string &message_name);
};

/**
 * User accessible interface to Spatial3D messages within mode description hierarchy
 * @see Data
 */
class MsgSpatial3D::Description : public MsgBruteForce::Description {
	/**
	* Data store class for this description, constructs instances of this class
	*/
	friend struct Data;

 protected:
	/**
	* Constructors
	*/
	Description(ModelData *const _model, Data *const data);
	/**
	* Default copy constructor, not implemented
	*/
	Description(const Description &other_message) = delete;
	/**
	* Default move constructor, not implemented
	*/
	Description(Description &&other_message) noexcept = delete;
	/**
	* Default copy assignment, not implemented
	*/
	Description& operator=(const Description &other_message) = delete;
	/**
	* Default move assignment, not implemented
	*/
	Description& operator=(Description &&other_message) noexcept = delete;

 public:
	void setRadius(const float &r);
	void setMinX(const float &x);
	void setMinY(const float &y);
	void setMinZ(const float &z);
	void setMin(const float &x, const float &y, const float &z);
	void setMaxX(const float &x);
	void setMaxY(const float &y);
	void setMaxZ(const float &z);
	void setMax(const float &x, const float &y, const float &z);

	float getRadius() const;
	float getMinX() const;
	float getMinY() const;
	float getMinZ() const;
	float getMaxX() const;
	float getMaxY() const;
	float getMaxZ() const;
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_SPATIAL3DHOST_H_
