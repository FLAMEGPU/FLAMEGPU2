// function.c // f_function.c
#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_
#include "handler.h"
#define radius 2.0f

int inputdata(AGENT_HANDLE *agent){ //__FLAME_GPU_FUNC__ int inputdata(xmachine_memory_Circle* xmemory, xmachine_message_location_list* location_messages){

const float kr = 0.1f; /* Stiffness variable for repulsion */
const float ka = 0.0f; /* Stiffness variable for attraction */

  float x1, y1, x2, y2, fx, fy;
  float location_distance, separation_distance;
  float k;

  x1 = getAgentVariable(agent,"x"); // x1 = xmemory->x;
  fx = 0.0;
  y1 = getAgentVariable(agent,"y"); // y1 = xmemory->y;
  fy = 0.0;

  /* Loop through all messages */
//xmachine_message_location* location_message = get_first_location_message(location_messages);
  MSG_HANDLE message_handle = getFirstMessage("location"); // returns the first message, we can use auto return types c++14

// may use iterator
  while(message_handle){
  if((getMessageVariable(message_handle,"id") != getAgentVariable(agent,"id"))){

  x2 = getMessageVariable(message_handle,"x"); // x2 = location_message->x;
  y2 = getMessageVariable(message_handle,"y"); // y2 = location_message->y;

  // Deep (expensive) check
  location_distance = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
  separation_distance = (location_distance - radius);

  if(separation_distance < radius){
  if(separation_distance > 0.0) k = ka;
  else k = -kr;

fx += k*(separation_distance)*((x1-x2)/radius);
fy += k*(separation_distance)*((y1-y2)/radius);
  }}

  /* Move onto next message to check */
//  location_message = get_next_location_message(location_message, location_messages);
  message_handle = getNextMessage("location");
  }
setAgentVariable(agent,"fx", fx); //  xmemory->fx = fx;
setAgentVariable(agent,"fy", fy); //  xmemory->fy = fy;


return 0;
}

int outputdata(AGENT_HANDLE *agent){
//__FLAME_GPU_FUNC__ int outputdata(xmachine_memory_Circle* xmemory, xmachine_message_location_list* location_messages){
  float x, y, z;

//x = xmemory->x; agent->getAgentVariable(AGENT_HANDLE, "x");
//y = xmemory->y; y= getAgentVariable(agent,"y");
//z = xmemory->z; z = getAgentVariable(agent,"z");

//message_handle = addMessage("location");
//setMessageVariable(message_handle, "x", 123);

// M: use variadic templates
addMessage("location", "id", getAgentVariable(agent,"id"),"x",getAgentVariable(agent,"x"), "y",getAgentVariable(agent,"y"), "z",getAgentVariable(agent,"z")); //add_location_message(location_messages, xmemory->id, x, y, z);

return 0;
}

//if doing kernel fusion, then we can have one function, but inside we specify the agent_handle.
int move (AGENT_HANDLE *agent){ // __FLAME_GPU_FUNC__ int move(xmachine_memory_Circle* xmemory)

x = getAgentVariable(agent,"x");
x =+ getAgentVariable(agent,"fx");
setAgentVariable(agent,"x",x); // xmemory->x += xmemory->fx;


y = getAgentVariable(agent,"y");
y += getAgentVariable(agent,"fy");
setAgentVariable(agent,"y", y); // xmemory->y += xmemory->fy;

return 0;
}
#endif // #ifndef _FUNCTIONS_H_
