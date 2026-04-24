#include "flamegpu/flamegpu.h"
#include "flamegpu/stockAgent/subModels/AddOneAgent.h"
#include <map>
#include <string>
#include <memory>

using namespace std;

namespace flamegpu {
namespace stockAgent {
namespace submodels {

const char* INTERNAL_AGENT_NAME = "movingAgent";

namespace {

    FLAMEGPU_AGENT_FUNCTION(movingAgent_broadcast_occupancy, MessageNone, MessageArray2D) {
        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");
        FLAMEGPU->message_out.setIndex(x, y);
        FLAMEGPU->message_out.setVariable<id_t>("occupant_id", FLAMEGPU->getID());
        return ALIVE;
    }

    FLAMEGPU_AGENT_FUNCTION(movingAgent_request_move, MessageArray2D, MessageSpatial2D) {
        if (FLAMEGPU->getVariable<int>("moved_this_step") == 1) {
            FLAMEGPU->setVariable<int>("target_x", -1);
            FLAMEGPU->setVariable<int>("target_y", -1);
        } else {
            int x = FLAMEGPU->getVariable<int>("x");
            int y = FLAMEGPU->getVariable<int>("y");

            int empty_count = 0;
            int empty_x[9];
            int empty_y[9];

            for (auto &msg : FLAMEGPU->message_in(x, y, 1)) {
                if (msg.getVariable<id_t>("occupant_id") == ID_NOT_SET) {
                    empty_x[empty_count] = (int)msg.getX();
                    empty_y[empty_count] = (int)msg.getY();
                    empty_count++;
                }
            }

            if (empty_count > 0) {
                int choice = FLAMEGPU->random.uniform<int>(0, empty_count - 1);
                int tx = empty_x[choice];
                int ty = empty_y[choice];
                FLAMEGPU->setVariable<int>("target_x", tx);
                FLAMEGPU->setVariable<int>("target_y", ty);
                
                FLAMEGPU->message_out.setVariable<id_t>("requester_id", FLAMEGPU->getID());
                FLAMEGPU->message_out.setVariable<float>("priority", FLAMEGPU->getVariable<float>("priority"));
                FLAMEGPU->message_out.setLocation(tx, ty);
            } else {
                FLAMEGPU->setVariable<int>("target_x", -1);
                FLAMEGPU->setVariable<int>("target_y", -1);
            }
        }

        return ALIVE;
    }

    FLAMEGPU_AGENT_FUNCTION(movingAgent_execute_move, MessageSpatial2D, MessageNone) {
        id_t my_id = FLAMEGPU->getID();
        int tx = FLAMEGPU->getVariable<int>("target_x");
        int ty = FLAMEGPU->getVariable<int>("target_y");

        if (tx != -1 && ty != -1) {
            float my_priority = FLAMEGPU->getVariable<float>("priority");
            bool won = true;

            for (auto &msg : FLAMEGPU->message_in(tx, ty)) {
                float other_priority = msg.getVariable<float>("priority");
                if (other_priority > my_priority) {
                    won = false;
                    break;
                } else if (other_priority == my_priority) {
                    if (msg.getVariable<id_t>("requester_id") > my_id) {
                        won = false;
                        break;
                    }
                }
            }

            if (won) {
                FLAMEGPU->setVariable<int>("x", tx);
                FLAMEGPU->setVariable<int>("y", ty);
                FLAMEGPU->setVariable<int>("moved_this_step", 1);
            }
        } 

        return ALIVE;
    }

    FLAMEGPU_HOST_CONDITION(move_exit_condition) {
        static int iterations = 0;
        iterations++;
        bool unresolved = FLAMEGPU->agent(INTERNAL_AGENT_NAME, "active").count<int>("moved_this_step", 0) > 0;
        if (unresolved && iterations < 5) {
            return CONTINUE;
        }
        iterations = 0;
        return EXIT;
    }

    FLAMEGPU_INIT_FUNCTION(reset_variables) {
        auto movingAgent = FLAMEGPU->agent(INTERNAL_AGENT_NAME, "active");
        auto &agent_pop = movingAgent.getPopulationData();
        for (auto agent : agent_pop) {
            agent.setVariable<int>("moved_this_step", 0);
            agent.setVariable<int>("target_x", -1);
            agent.setVariable<int>("target_y", -1);
        }
    }
}

SubModelDescription AddOneAgent::addOneAgentSubmodel(ModelDescription &model, int ENV_SIZE_X, int ENV_SIZE_Y) {
    ModelDescription sub_model_move("movement_submodel");

    AgentDescription agent = sub_model_move.newAgent(INTERNAL_AGENT_NAME);
    agent.newState("active");
    agent.newState("stationary");
    agent.newVariable<int>("x");
    agent.newVariable<int>("y");
    agent.newVariable<float>("priority", 0.0f);
    agent.newVariable<int>("target_x", -1);
    agent.newVariable<int>("target_y", -1);
    agent.newVariable<int>("moved_this_step", 0);

    defineMessageSubmodule(sub_model_move, ENV_SIZE_X, ENV_SIZE_Y);
    setMessages(agent, ENV_SIZE_X, ENV_SIZE_Y);
    defineLayer(sub_model_move);
    sub_model_move.addInitFunction(reset_variables);
    sub_model_move.addExitCondition(move_exit_condition);

    // Initialize the internal handle
    this->smm = std::make_unique<SubModelDescription>(model.newSubModel("MovementInstance", sub_model_move));
    this->smm->setMaxSteps(1);
    this->is_initialized = true;

    return *(this->smm);
}

void AddOneAgent::setAgent(const string& parent_name,
                           const map<string, string>& var_map,
                           const map<string, string>& state_map,
                           bool auto_map) {
    if (!is_initialized) {
        throw exception::InvalidSubModel("AddOneAgent submodel was not initialized. Call addOneAgentSubmodel() first.");
    }

    auto agent_map = this->smm->bindAgent(INTERNAL_AGENT_NAME, parent_name, auto_map, auto_map);
    
    for (auto const& [internal_var, parent_var] : var_map) {
        agent_map.mapVariable(internal_var, parent_var);
    }

    for (auto const& [internal_state, parent_state] : state_map) {
        agent_map.mapState(internal_state, parent_state);
    }

    this->validate();
}

void AddOneAgent::defineMessageSubmodule(ModelDescription &smm, int ENV_SIZE_X, int ENV_SIZE_Y) {
    auto m1 = smm.newMessage<MessageArray2D>("occupancy_status");
    m1.newVariable<id_t>("occupant_id");
    m1.setDimensions(ENV_SIZE_X, ENV_SIZE_Y);

    auto m2 = smm.newMessage<MessageSpatial2D>("move_requests");
    m2.newVariable<id_t>("requester_id");
    m2.newVariable<float>("priority");
    m2.setMin(0, 0);
    m2.setMax(ENV_SIZE_X, ENV_SIZE_Y);
    m2.setRadius(0.1f);
}

void AddOneAgent::setMessages(AgentDescription &movingAgent, int ENV_SIZE_X, int ENV_SIZE_Y) {
    auto f1 = movingAgent.newFunction("movingAgent_broadcast_occupancy", movingAgent_broadcast_occupancy);
    f1.setInitialState("active");
    f1.setEndState("active");
    f1.setMessageOutput("occupancy_status");
    
    auto f2 = movingAgent.newFunction("movingAgent_request_move", movingAgent_request_move);
    f2.setInitialState("active");
    f2.setEndState("active");
    f2.setMessageInput("occupancy_status");
    f2.setMessageOutput("move_requests");
    
    auto f3 = movingAgent.newFunction("movingAgent_execute_move", movingAgent_execute_move);
    f3.setInitialState("active");
    f3.setEndState("active");
    f3.setMessageInput("move_requests");
}

void AddOneAgent::defineLayer(ModelDescription &smm) {
    smm.newLayer().addAgentFunction(movingAgent_broadcast_occupancy);
    smm.newLayer().addAgentFunction(movingAgent_request_move);
    smm.newLayer().addAgentFunction(movingAgent_execute_move);
}

void AddOneAgent::validate() {
    if (!this->is_initialized) {
        throw exception::InvalidSubModel("AddOneAgent submodel was not initialized. Call addOneAgentSubmodel() first.");
    }

    try {
        auto sub_agent = this->smm->getSubAgent(INTERNAL_AGENT_NAME);
        
        // Mandatory mappings check
        sub_agent.getVariableMapping("x");
        sub_agent.getVariableMapping("y");
        sub_agent.getVariableMapping("priority");

    } catch (const exception::InvalidSubAgentName& e) {
        string msg = "AddOneAgent submodel missing required agent binding: " + string(e.what());
        throw exception::InvalidSubAgentName(msg.c_str());
    } catch (const exception::InvalidAgentVar& e) {
        string msg = "AddOneAgent submodel missing required variable mapping: " + string(e.what());
        throw exception::InvalidAgentVar(msg.c_str());
    }
}

} // namespace submodels
} // namespace stockAgent
} // namespace flamegpu
