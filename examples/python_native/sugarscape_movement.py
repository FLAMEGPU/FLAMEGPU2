import pyflamegpu
import pyflamegpu.codegen
import sugarscape_common

def create_submodel(model):
    movement_model = pyflamegpu.ModelDescription("movement-model")
    # link the two models
    submodel = model.newSubModel("movement-submodel", movement_model)
    return movement_model, submodel

def define_submodel_environment(movement_model):
    env = movement_model.Environment()
    # agent status
    env.newPropertyInt("agent_status_unoccupied", sugarscape_common.AGENT_STATUS_UNOCCUPIED)
    env.newPropertyInt("agent_status_occupied", sugarscape_common.AGENT_STATUS_OCCUPIED)
    env.newPropertyInt("agent_status_movement_requested", sugarscape_common.AGENT_STATUS_MOVEMENT_REQUESTED)
    env.newPropertyInt("agent_status_movement_unresolved", sugarscape_common.AGENT_STATUS_MOVEMENT_UNRESOLVED)

def define_submodel_messages(movement_model, env_size):
    """
    Cell status message
    """      
    message = movement_model.newMessageArray2D("cell_status")
    message.newVariableID("location_id")
    message.newVariableInt("status")
    message.newVariableInt("env_sugar_level")
    message.setDimensions(env_size, env_size)

    """
    Movement request message
    """      
    message = movement_model.newMessageArray2D("movement_request")
    message.newVariableID("location_id")
    message.newVariableID("agent_id")
    message.newVariableInt("sugar_level")
    message.newVariableInt("metabolism")
    message.setDimensions(env_size, env_size)

    """
    Movement response message
    """      
    message = movement_model.newMessageArray2D("movement_response")
    # message.newVariableID("location_id")
    message.newVariableID("agent_id")
    message.setDimensions(env_size, env_size)



def define_submodel_agents(movement_model):
    agent = movement_model.newAgent("agent")

    # Generic variables
    agent.newVariableInt("x")
    agent.newVariableInt("y")
    agent.newVariableInt("agent_id")
    agent.newVariableInt("status")
    # Sugarscape agent specific variables
    agent.newVariableInt("sugar_level")
    agent.newVariableInt("metabolism")
    # Environment specific variables
    agent.newVariableInt("env_sugar_level")
    agent.newVariableInt("env_max_sugar_level")
    
    # Output cell status function
    fn = agent.newRTCFunction("output_cell_status", pyflamegpu.codegen.translate(output_cell_status))
    fn.setMessageOutput("cell_status")

    # Movement request function
    fn = agent.newRTCFunction("movement_request", pyflamegpu.codegen.translate(movement_request))
    fn.setMessageInput("cell_status")
    fn.setMessageOutput("movement_request")

    # Movement response function
    fn = agent.newRTCFunction("movement_response", pyflamegpu.codegen.translate(movement_response))
    fn.setMessageInput("movement_request")
    fn.setMessageOutput("movement_response")

    # Movement Transaction
    fn = agent.newRTCFunction("movement_transaction", pyflamegpu.codegen.translate(movement_transaction))
    fn.setMessageInput("movement_response")

def define_submodel_execution_order(movement_model):
    """
      Control flow
    """    
    movement_model.newLayer().addAgentFunction("agent", "output_cell_status")
    movement_model.newLayer().addAgentFunction("agent", "movement_request")
    movement_model.newLayer().addAgentFunction("agent", "movement_response")
    movement_model.newLayer().addAgentFunction("agent", "movement_transaction")
    movement_model.addExitCondition(MovementExitCondition());


@pyflamegpu.agent_function
def output_cell_status(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageArray2D):
    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")

    message_out.setIndex(agent_x, agent_y)
    message_out.setVariableInt("location_id", pyflamegpu.getID())
    message_out.setVariableInt("status", pyflamegpu.getVariableInt("status"))
    message_out.setVariableInt("env_sugar_level", pyflamegpu.getVariableInt("env_sugar_level"))

    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def movement_request(message_in: pyflamegpu.MessageArray2D, message_out: pyflamegpu.MessageArray2D):
    best_sugar_level = -1
    best_sugar_random = float(-1)
    best_location_id = 0

    # if occupied then look for empty cells {
    # find the best location to move to (ensure we don't just pick first cell with max value)
    status = pyflamegpu.getVariableInt("status")

    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")

    # if occupied then look for empty cells
    if (status == pyflamegpu.environment.getPropertyInt("agent_status_movement_unresolved")) :
        for current_message in message_in.wrap(agent_x, agent_y) :
            # if location is unoccupied then check for empty locations
            if (current_message.getVariableInt("status") == pyflamegpu.environment.getPropertyInt("agent_status_unoccupied")) :
                # if the sugar level at current location is better than currently stored then update
                message_env_sugar_level = current_message.getVariableInt("env_sugar_level")
                message_priority = pyflamegpu.random.uniformFloat()
                if ((message_env_sugar_level >  best_sugar_level)   or
                    (message_env_sugar_level == best_sugar_level    and 
                     message_priority        >  best_sugar_random)) :
                    best_sugar_level = message_env_sugar_level
                    best_sugar_random = message_priority
                    best_location_id = current_message.getVariableInt("location_id")

        # if the agent has found a better location to move to then update its state
        # if there is a better location to move to then state indicates a movement request
        if best_location_id != 0 :
            status = pyflamegpu.environment.getPropertyInt("agent_status_movement_requested") 
        else : 
            status = pyflamegpu.environment.getPropertyInt("agent_status_occupied")            
        pyflamegpu.setVariableInt("status", status)

    # add a movement request
    message_out.setIndex(agent_x, agent_y)
    message_out.setVariableInt("agent_id", pyflamegpu.getVariableInt("agent_id"))
    message_out.setVariableInt("location_id", best_location_id)
    message_out.setVariableInt("sugar_level", pyflamegpu.getVariableInt("sugar_level"))
    message_out.setVariableInt("metabolism", pyflamegpu.getVariableInt("metabolism"))

    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def movement_response(message_in: pyflamegpu.MessageArray2D, message_out: pyflamegpu.MessageArray2D):
    best_request_id = -1
    best_request_priority = float(-1)
    best_request_sugar_level = -1
    best_request_metabolism = -1

    status = pyflamegpu.getVariableInt("status")
    location_id = pyflamegpu.getID()
    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")

    for current_message in message_in.wrap(agent_x, agent_y) :
        # if the location is unoccupied then check for agents requesting to move here
        if (status == pyflamegpu.environment.getPropertyInt("agent_status_unoccupied")) :
            # check if request is to move to this location
            if (current_message.getVariableInt("location_id") == location_id) :
                # check the priority and maintain the best ranked agent
                message_priority = pyflamegpu.random.uniformFloat()
                if (message_priority > best_request_priority) :
                    best_request_id = current_message.getVariableInt("agent_id")
                    best_request_priority = message_priority
                    best_request_sugar_level = current_message.getVariableInt("sugar_level")
                    best_request_metabolism = current_message.getVariableInt("metabolism")

    # if the location is unoccupied and an agent wants to move here then do so and send a response
    if ((status == pyflamegpu.environment.getPropertyInt("agent_status_unoccupied")) and (best_request_id >= 0)) :
        pyflamegpu.setVariableInt("status", pyflamegpu.environment.getPropertyInt("agent_status_occupied"))
        # move the agent to here 
        pyflamegpu.setVariableInt("agent_id", best_request_id)
        pyflamegpu.setVariableInt("sugar_level", best_request_sugar_level)
        pyflamegpu.setVariableInt("metabolism", best_request_metabolism)

    # add a movement response
    message_out.setIndex(agent_x, agent_y)
    message_out.setVariableInt("agent_id", best_request_id)

    return pyflamegpu.ALIVE


@pyflamegpu.agent_function
def movement_transaction(message_in: pyflamegpu.MessageArray2D, message_out: pyflamegpu.MessageNone):
    status = pyflamegpu.getVariableInt("status")
    agent_id = pyflamegpu.getVariableInt("agent_id")
    agent_x = pyflamegpu.getVariableInt("x")
    agent_y = pyflamegpu.getVariableInt("y")

    for current_message in message_in.wrap(agent_x, agent_y) :
        # if location contains an agent wanting to move then look for responses allowing relocation
        if (status == pyflamegpu.environment.getPropertyInt("agent_status_movement_requested")) :  
            # if the movement response request came from this location
            if (current_message.getVariableInt("agent_id") == agent_id) :
                # remove the agent and reset agent specific variables as it has now moved
                status = pyflamegpu.environment.getPropertyInt("agent_status_unoccupied")
                pyflamegpu.setVariableInt("agent_id", -1)
                pyflamegpu.setVariableInt("sugar_level", 0)
                pyflamegpu.setVariableInt("metabolism", 0)


    # if request has not been responded to then agent is unresolved
    if (status == pyflamegpu.environment.getPropertyInt("agent_status_movement_requested")) :
        status = pyflamegpu.environment.getPropertyInt("agent_status_movement_unresolved")

    pyflamegpu.setVariableInt("status", status)

    return pyflamegpu.ALIVE


# Define a host condition function
class MovementExitCondition(pyflamegpu.HostCondition):

    def __init__(self):
        super().__init__()
        self.iterations = 0
    
    def run(self, FLAMEGPU):
        self.iterations += 1
        # Max iterations 9
        if (self.iterations < 9) :
            # Agent movements still unresolved
            if FLAMEGPU.agent("agent").countInt("status", sugarscape_common.AGENT_STATUS_MOVEMENT_UNRESOLVED) :
                return pyflamegpu.CONTINUE   
            # Helpful debug statement to see how many iterations are required to resolve the conflicts
            # print(f"Conflict resolved by iteration {self.iterations}")
        self.iterations = 0
        return pyflamegpu.EXIT

def add_movement_submodel(model, env_size):
    movement_model, submodel = create_submodel(model)
    
    define_submodel_environment(movement_model)
    define_submodel_messages(movement_model, env_size)
    define_submodel_agents(movement_model)
    define_submodel_execution_order(movement_model)

    # Note: submodel must be added to a dependency graph
    
    submodel.bindAgent("agent", "agent", True, True);

    return submodel

    