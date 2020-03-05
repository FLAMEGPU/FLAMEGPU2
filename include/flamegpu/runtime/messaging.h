#ifndef INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
#define INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_

/**
 * This file provides the root include for messaging
 * The bulk of message specialisation is implemented within headers included at the bottom of this file
 */

#include "flamegpu/runtime/messaging/None.h"
#include "flamegpu/runtime/messaging/BruteForce.h"
#include "flamegpu/runtime/messaging/Spatial2D.h"
#include "flamegpu/runtime/messaging/Spatial3D.h"
#include "flamegpu/runtime/messaging/Array.h"
#include "flamegpu/runtime/messaging/Array2D.h"
#include "flamegpu/runtime/messaging/Array3D.h"

/**
 * ######################################################
 * # Summary of how new message types should be defined #
 * ######################################################
 * 
 * Each message type is defined as a group of required nested classes
 * these are then used via C++ templating, therefore they require some
 * common components
 * 
 * Nested Classes:
 *  
 *  In:
 *  This is available to the user via the FLAMEGPU_DEVICE_API object passed into agent functions.
 *  This class should provide users access to messages in the message list.
 *  
 *  It's only required methods are the format of the construtor, see None.h for example.
 *  
 *  Out:
 *  This is available to the user via the FLAMEGPU_DEVICE_API object passed into agent functions.
 *  This class should provide users access to messages ouput functionality
 *  
 *  It's only required methods are the format of the construtor, see None.h for example.
 *  
 *  CUDAModelHandler:
 *  This is used internally, with one allocated inside each CUDAMessage created by CUDAAgentModel.
 *  This class provides message specific handling of message lists, 
 *  some of this functionality may not be required for your messaging type.
 *  
 *  It is required to have the correct constructor format, and to inherit from MsgSpecialisationHandler.
 *  
 *  The method buildIndex() is called the first time messages are read after message output. This
 *  is useful if your messaging type required a special index (e.g. Spatial messaging PBM). Read list 
 *  (d_list) must contain the sorted message data on exit from the method.
 *  
 *  The method getMetaDataDevicePtr() returns a pointer to a structure on the device that is required for
 *  message input. This pointer is then passed to the constructor of In. This is useful if your messaging
 *  data structure requires additional metadata.
 *  
 *  Description:
 *  This is the class a user interacts with to configure their message as part of a ModelDescription
 *  hierarchy. It is expected to follow the same style as all the existing Description classes. 
 *  
 *  This class will likely inherit from MsgBruteForce::Description.
 *  
 *  Data:
 *  This is the class that stores the data behind the scenes of the ModelDescription hierarchy.
 *  It is expected to follow the same style as all the existing Data classes. 
 *  It must have a functional copy constructor and equality operators. However, due to the hierarchical
 *  nature the protoype for the copy constructor takes some additional arguments.
 *  
 *  This class will likely inherit from MsgBruteForce::Description.
 *  
 */

#endif  // INCLUDE_FLAMEGPU_RUNTIME_MESSAGING_H_
