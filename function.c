/******************************************************************************
* This is an example "function.c" file
******************************************************************************
* Author  Mozhgan Kabiri Chimeh
* Date    
*****************************************************************************/

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_
#include "handler.h"
#define radius 2.0f

#include "runtime/flame_api.h"

FLAMEGPU_AGENT_FUNCTION(inputdata) {

	const float kr = 0.1f; /* Stiffness variable for repulsion */
	const float ka = 0.0f; /* Stiffness variable for attraction */

	float x1, y1, x2, y2, fx, fy;
	float location_distance, separation_distance;
	float k;

	x1 = FLAMEGPU->getVariable<float>("x");
	fx = 0.0;
	y1 = FLAMEGPU->getVariable<float>("y");
	fy = 0.0;

	/* Loop through all messages */
  //xmachine_message_location* location_message = get_first_location_message(location_messages);
	MSG_HANDLE message_handle = getFirstMessage("location"); // returns the first message, we can use auto return types c++14

  // may use iterator
	while (message_handle) {
		if (FLAMEGPU->getMessageVariable("id") != FLAMEGPU->getVariable("id")) {

			x2 = FLAMEGPU->getMessageVariable<float>("x"); // x2 = location_message->x;
			y2 = FLAMEGPU->getMessageVariable<float>("y"); // y2 = location_message->y;

			// Deep (expensive) check
			location_distance = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
			separation_distance = (location_distance - radius);

			if (separation_distance < radius) {
				if (separation_distance > 0.0) k = ka;
				else k = -kr;

				fx += k*(separation_distance)*((x1 - x2) / radius);
				fy += k*(separation_distance)*((y1 - y2) / radius);
			}
		}

		/* Move onto next message to check */
	  //  location_message = get_next_location_message(location_message, location_messages);
		message_handle = getNextMessage("location");
	}
	FLAMEGPU->setVariable<float>("fx", fx);
	FLAMEGPU->setVariable<float>("fy", fy);


	return 0;
}

FLAMEGPU_AGENT_FUNCTION(outputdata) {
	
	//message_handle = addMessage("location");
	//setMessageVariable(message_handle, "x", 123);

	// M: use variadic templates?
	addMessage("location", "id", FLAMEGPU->getVariable<float>("id"), "x", FLAMEGPU->getVariable<float>("x"), "y", FLAMEGPU->getVariable<float>("y"), "z", FLAMEGPU->getVariable<float>("z")); //add_location_message(location_messages, xmemory->id, x, y, z);

	return 0;
}

//if doing kernel fusion, then we can have one function, but inside we specify the agent_handle.
FLAMEGPU_AGENT_FUNCTION(move) {

	x = FLAMEGPU->getVariable<float>( "x");
	x =+FLAMEGPU->getVariable<float>( "fx");
	FLAMEGPU->setVariable<float>("x", x);


	y = FLAMEGPU->getVariable<float>("y");
	y += FLAMEGPU->getVariable<float>("fy");
	FLAMEGPU->setVariable<float>("y", y);

	return 0;
}
#endif // #ifndef _FUNCTIONS_H_
