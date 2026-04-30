#include "flamegpu/stockAgent/subModels/SingleAgentDiscreteMovement.h"

#include <map>
#include <string>
#include <memory>
#include <vector>

#include "flamegpu/flamegpu.h"

using std::string;
using std::map;

namespace flamegpu {
namespace stockAgent {
namespace submodels {

const char* INTERNAL_MOVING_AGENT_NAME = "MovingAgent";
const char* INTERNAL_ENV_AGENT_NAME = "GridCell";

namespace {

    /**
     * 1. Environment agent broadcasts its status (score/nectar and current occupancy)
     */
    FLAMEGPU_AGENT_FUNCTION(envAgent_broadcast_status, MessageNone, MessageArray2D) {
        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");
        int is_occupied = FLAMEGPU->getVariable<int>("is_occupied");
        float score = FLAMEGPU->getVariable<float>("cell_score");

        FLAMEGPU->message_out.setIndex(x, y);
        FLAMEGPU->message_out.setVariable<int>("is_occupied", is_occupied);
        FLAMEGPU->message_out.setVariable<float>("cell_score", score);
        return ALIVE;
    }

    /**
     * 2. Moving agent looks at neighbors and requests a move to the best UNOCCUPIED one.
     */
    FLAMEGPU_AGENT_FUNCTION(movingAgent_request_move, MessageArray2D, MessageArray2D) {
        // Reset movement status ONLY on the first internal iteration of the submodel call.
        if (FLAMEGPU->getStepCounter() == 0) {
            FLAMEGPU->setVariable<int>("moved_this_step", 0);
            FLAMEGPU->setVariable<int>("target_x", -1);
            FLAMEGPU->setVariable<int>("target_y", -1);
        }

        if (FLAMEGPU->getVariable<int>("moved_this_step") == 1) {
            // Always output a message to keep the MessageArray2D dense/occupied at current location
            FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<int>("x"), FLAMEGPU->getVariable<int>("y"));
            FLAMEGPU->message_out.setVariable<id_t>("requester_id", FLAMEGPU->getID());
            FLAMEGPU->message_out.setVariable<int>("target_x", -1);
            FLAMEGPU->message_out.setVariable<int>("target_y", -1);
            FLAMEGPU->message_out.setVariable<float>("priority", -1.0f);
            return ALIVE;
        }

        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");
        int lx = FLAMEGPU->getVariable<int>("last_x");
        int ly = FLAMEGPU->getVariable<int>("last_y");
        int lfx = FLAMEGPU->getVariable<int>("last_resources_x");
        int lfy = FLAMEGPU->getVariable<int>("last_resources_y");
        float current_cell_score = FLAMEGPU->getVariable<float>("current_cell_score");

        float best_neighbor_score = -1e10f;
        float max_tie_breaker = -1.0f;
        int target_x = -1;
        int target_y = -1;

        for (auto &msg : FLAMEGPU->message_in(x, y, 1)) {
            int mx = msg.getX();
            int my = msg.getY();

            // Consider only unoccupied cells
            // AND exclude the cell we just came from (lx, ly)
            // AND exclude the last resource we visited (lfx, lfy)
            if (msg.getVariable<int>("is_occupied") == 0 && !(mx == lx && my == ly) && !(mx == lfx && my == lfy)) {
                float n = msg.getVariable<float>("cell_score");
                float tie_breaker = FLAMEGPU->random.uniform<float>();

                if (n > best_neighbor_score || (n == best_neighbor_score && tie_breaker > max_tie_breaker)) {
                    best_neighbor_score = n;
                    max_tie_breaker = tie_breaker;
                    target_x = mx;
                    target_y = my;
                }
            }
        }

        // Only request a move if the best neighbor is at least as good as current spot
        if (target_x != -1 && best_neighbor_score >= current_cell_score) {
            FLAMEGPU->setVariable<int>("target_x", target_x);
            FLAMEGPU->setVariable<int>("target_y", target_y);
        } else {
            FLAMEGPU->setVariable<int>("target_x", -1);
            FLAMEGPU->setVariable<int>("target_y", -1);
        }

        FLAMEGPU->message_out.setIndex(x, y);
        FLAMEGPU->message_out.setVariable<id_t>("requester_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("target_x", FLAMEGPU->getVariable<int>("target_x"));
        FLAMEGPU->message_out.setVariable<int>("target_y", FLAMEGPU->getVariable<int>("target_y"));
        FLAMEGPU->message_out.setVariable<float>("priority", FLAMEGPU->getVariable<float>("priority"));

        return ALIVE;
    }

    /**
     * 3. Environment agent resolves conflicts.
     */
    FLAMEGPU_AGENT_FUNCTION(envAgent_resolve_conflict, MessageArray2D, MessageArray2D) {
        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");

        id_t winner_id = ID_NOT_SET;
        float best_priority = -1e10f;
        float max_tie_breaker = -1.0f;

        // Only resolve if currently unoccupied
        if (FLAMEGPU->getVariable<int>("is_occupied") == 0) {
            for (auto &msg : FLAMEGPU->message_in(x, y, 1)) {
                if (msg.getVariable<int>("target_x") == x && msg.getVariable<int>("target_y") == y) {
                    float p = msg.getVariable<float>("priority");
                    float tie_breaker = FLAMEGPU->random.uniform<float>();

                    if (p > best_priority || (p == best_priority && tie_breaker > max_tie_breaker)) {
                        best_priority = p;
                        max_tie_breaker = tie_breaker;
                        winner_id = msg.getVariable<id_t>("requester_id");
                    }
                }
            }
        }

        FLAMEGPU->message_out.setIndex(x, y);
        FLAMEGPU->message_out.setVariable<id_t>("winner_id", winner_id);
        FLAMEGPU->message_out.setVariable<float>("cell_score", FLAMEGPU->getVariable<float>("cell_score"));
        return ALIVE;
    }

    /**
     * 4. Moving agent checks if it won the cell it requested.
     */
    FLAMEGPU_AGENT_FUNCTION(movingAgent_execute_move, MessageArray2D, MessageArray2D) {
        int tx = FLAMEGPU->getVariable<int>("target_x");
        int ty = FLAMEGPU->getVariable<int>("target_y");
        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");
        id_t my_id = FLAMEGPU->getID();

        if (tx != -1 && ty != -1) {
            auto msg = FLAMEGPU->message_in.at(tx, ty);
            if (msg.getVariable<id_t>("winner_id") == my_id) {
                // Update "last" position before moving
                FLAMEGPU->setVariable<int>("last_x", x);
                FLAMEGPU->setVariable<int>("last_y", y);

                x = tx;
                y = ty;
                FLAMEGPU->setVariable<int>("x", x);
                FLAMEGPU->setVariable<int>("y", y);
                FLAMEGPU->setVariable<int>("moved_this_step", 1);

                // If the new cell has resources, update last_resources
                float score = msg.getVariable<float>("cell_score");
                if (score > 0.01f) {
                    FLAMEGPU->setVariable<int>("last_resources_x", x);
                    FLAMEGPU->setVariable<int>("last_resources_y", y);
                }
            }
        }

        // Always sample current cell score
        auto msg = FLAMEGPU->message_in.at(x, y);
        FLAMEGPU->setVariable<float>("current_cell_score", msg.getVariable<float>("cell_score"));

        // Notify current location of presence
        FLAMEGPU->message_out.setIndex(x, y);
        FLAMEGPU->message_out.setVariable<id_t>("moving_agent_id", my_id);

        return ALIVE;
    }

    /**
     * 5. Environment agent updates its occupancy status.
     */
    FLAMEGPU_AGENT_FUNCTION(envAgent_update_occupancy, MessageArray2D, MessageNone) {
        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");
        auto msg = FLAMEGPU->message_in.at(x, y);
        FLAMEGPU->setVariable<int>("is_occupied", (msg.getVariable<id_t>("moving_agent_id") != ID_NOT_SET) ? 1 : 0);
        return ALIVE;
    }

    FLAMEGPU_HOST_CONDITION(move_exit_condition) {
        static int iterations = 0;
        iterations++;
        bool unresolved = FLAMEGPU->agent(INTERNAL_MOVING_AGENT_NAME, "active").count<int>("moved_this_step", 0) > 0;
        if (unresolved && iterations < 5) {
            return CONTINUE;
        }
        iterations = 0;
        return EXIT;
    }

    FLAMEGPU_INIT_FUNCTION(reset_variables) {
        auto movingAgent = FLAMEGPU->agent(INTERNAL_MOVING_AGENT_NAME, "active");
        auto &movingPopulation = movingAgent.getPopulationData();
        for (auto agent : movingPopulation) {
            agent.setVariable<int>("moved_this_step", 0);
            agent.setVariable<int>("target_x", -1);
            agent.setVariable<int>("target_y", -1);
        }
    }
}  // namespace

flamegpu::SubModelDescription SingleAgentDiscreteMovement::addSingleAgentDiscreteMovementSubmodel(ModelDescription &model, int ENV_SIZE_X, int ENV_SIZE_Y) {
    ModelDescription sub_model_move("movement_submodel");

    sub_model_move.Environment().newProperty<int>("submodel_env_width", ENV_SIZE_X);
    sub_model_move.Environment().newProperty<int>("submodel_env_height", ENV_SIZE_Y);


    AgentDescription movingAgent = sub_model_move.newAgent(INTERNAL_MOVING_AGENT_NAME);
    movingAgent.newState("active");
    movingAgent.newVariable<int>("x");
    movingAgent.newVariable<int>("y");
    movingAgent.newVariable<float>("priority", 0.0f);
    movingAgent.newVariable<float>("current_cell_score", 0.0f);
    movingAgent.newVariable<int>("target_x", -1);
    movingAgent.newVariable<int>("target_y", -1);
    movingAgent.newVariable<int>("moved_this_step", 0);
    movingAgent.newVariable<int>("last_x", -1);
    movingAgent.newVariable<int>("last_y", -1);
    movingAgent.newVariable<int>("last_resources_x", -1);
    movingAgent.newVariable<int>("last_resources_y", -1);

    AgentDescription envAgent = sub_model_move.newAgent(INTERNAL_ENV_AGENT_NAME);
    envAgent.newState("active");
    envAgent.newVariable<int>("x");
    envAgent.newVariable<int>("y");
    envAgent.newVariable<int>("is_occupied", 0);
    envAgent.newVariable<float>("cell_score", 0.0f);

    defineMessageSubmodule(sub_model_move, ENV_SIZE_X, ENV_SIZE_Y);
    setMessages(movingAgent, envAgent, ENV_SIZE_X, ENV_SIZE_Y);
    defineLayer(sub_model_move);
    sub_model_move.addInitFunction(reset_variables);
    sub_model_move.addExitCondition(move_exit_condition);

    this->smm = std::make_unique<SubModelDescription>(model.newSubModel("MovementInstance", sub_model_move));
    this->smm->setMaxSteps(5);
    this->is_initialized = true;

    return *(this->smm);
}

void SingleAgentDiscreteMovement::setMovingAgent(const string& parent_name,
                           const map<string, string>& var_map,
                           const map<string, string>& state_map,
                           bool auto_map) {
    if (!is_initialized) {
        throw exception::InvalidSubModel("SingleAgentDiscreteMovement submodel was not initialized. Call addSingleAgentDiscreteMovementSubmodel() first.");
    }
    auto agent_map = this->smm->bindAgent(INTERNAL_MOVING_AGENT_NAME, parent_name, auto_map, auto_map);
    for (auto const& [internal_var, parent_var] : var_map) { agent_map.mapVariable(internal_var, parent_var); }
    for (auto const& [internal_state, parent_state] : state_map) { agent_map.mapState(internal_state, parent_state); }
}

void SingleAgentDiscreteMovement::setEnvironmentAgent(const string& parent_name,
                           const map<string, string>& var_map,
                           const map<string, string>& state_map,
                           bool auto_map) {
    if (!is_initialized) {
        throw exception::InvalidSubModel("SingleAgentDiscreteMovement submodel was not initialized. Call addSingleAgentDiscreteMovementSubmodel() first.");
    }
    auto agent_map = this->smm->bindAgent(INTERNAL_ENV_AGENT_NAME, parent_name, auto_map, auto_map);
    for (auto const& [internal_var, parent_var] : var_map) { agent_map.mapVariable(internal_var, parent_var); }
    for (auto const& [internal_state, parent_state] : state_map) { agent_map.mapState(internal_state, parent_state); }
}

void SingleAgentDiscreteMovement::defineMessageSubmodule(ModelDescription &smm, int ENV_SIZE_X, int ENV_SIZE_Y) {
    auto m1 = smm.newMessage<MessageArray2D>("cell_status");
    m1.newVariable<float>("cell_score");
    m1.newVariable<int>("is_occupied");
    m1.setDimensions(ENV_SIZE_X, ENV_SIZE_Y);

    auto m2 = smm.newMessage<MessageArray2D>("move_requests");
    m2.newVariable<id_t>("requester_id");
    m2.newVariable<int>("target_x");
    m2.newVariable<int>("target_y");
    m2.newVariable<float>("priority");
    m2.setDimensions(ENV_SIZE_X, ENV_SIZE_Y);

    auto m3 = smm.newMessage<MessageArray2D>("move_responses");
    m3.newVariable<id_t>("winner_id");
    m3.newVariable<float>("cell_score");
    m3.setDimensions(ENV_SIZE_X, ENV_SIZE_Y);

    auto m4 = smm.newMessage<MessageArray2D>("new_locations");
    m4.newVariable<id_t>("moving_agent_id");
    m4.setDimensions(ENV_SIZE_X, ENV_SIZE_Y);
}

void SingleAgentDiscreteMovement::setMessages(AgentDescription &movingAgent, AgentDescription &envAgent, int ENV_SIZE_X, int ENV_SIZE_Y) {
    envAgent.newFunction("envAgent_broadcast_status", envAgent_broadcast_status).setMessageOutput("cell_status");

    auto movingAgent_request_move_fn = movingAgent.newFunction("movingAgent_request_move", movingAgent_request_move);
    movingAgent_request_move_fn.setMessageInput("cell_status");
    movingAgent_request_move_fn.setMessageOutput("move_requests");

    auto envAgent_resolve_conflict_fn = envAgent.newFunction("envAgent_resolve_conflict", envAgent_resolve_conflict);
    envAgent_resolve_conflict_fn.setMessageInput("move_requests");
    envAgent_resolve_conflict_fn.setMessageOutput("move_responses");

    auto movingAgent_execute_move_fn = movingAgent.newFunction("movingAgent_execute_move", movingAgent_execute_move);
    movingAgent_execute_move_fn.setMessageInput("move_responses");
    movingAgent_execute_move_fn.setMessageOutput("new_locations");

    auto envAgent_update_occupancy_fn = envAgent.newFunction("envAgent_update_occupancy", envAgent_update_occupancy);
    envAgent_update_occupancy_fn.setMessageInput("new_locations");
}

void SingleAgentDiscreteMovement::defineLayer(ModelDescription &smm) {
    smm.newLayer().addAgentFunction(envAgent_broadcast_status);
    smm.newLayer().addAgentFunction(movingAgent_request_move);
    smm.newLayer().addAgentFunction(envAgent_resolve_conflict);
    smm.newLayer().addAgentFunction(movingAgent_execute_move);
    smm.newLayer().addAgentFunction(envAgent_update_occupancy);
}

void SingleAgentDiscreteMovement::validate() {
    if (!this->is_initialized) {
        throw exception::InvalidSubModel("SingleAgentDiscreteMovement submodel was not initialized. Call addSingleAgentDiscreteMovementSubmodel() first.");
    }

    try {
        auto moving_agent = this->smm->getSubAgent(INTERNAL_MOVING_AGENT_NAME);
        auto env_agent = this->smm->getSubAgent(INTERNAL_ENV_AGENT_NAME);

        // Mandatory mappings check
        moving_agent.getVariableMapping("x");
        moving_agent.getVariableMapping("y");
        moving_agent.getVariableMapping("last_x");
        moving_agent.getVariableMapping("last_y");
        moving_agent.getVariableMapping("last_resources_x");
        moving_agent.getVariableMapping("last_resources_y");
        moving_agent.getVariableMapping("current_cell_score");

        env_agent.getVariableMapping("x");
        env_agent.getVariableMapping("y");
        env_agent.getVariableMapping("is_occupied");
        env_agent.getVariableMapping("cell_score");

        // Mandatory state mapping check
        env_agent.getStateMapping("active");
        moving_agent.getStateMapping("active");
    } catch (const exception::InvalidAgentState& e) {
        string msg = "SingleAgentDiscreteMovement submodel missing required state binding: " + string(e.what());
        throw exception::InvalidAgentState(msg.c_str());
    } catch (const exception::InvalidSubAgentName& e) {
        string msg = "SingleAgentDiscreteMovement submodel missing required agent binding: " + string(e.what());
        throw exception::InvalidSubAgentName(msg.c_str());
    } catch (const exception::InvalidAgentVar& e) {
        string msg = "SingleAgentDiscreteMovement submodel missing required variable mapping: " + string(e.what());
        throw exception::InvalidAgentVar(msg.c_str());
    }
}

}  // namespace submodels
}  // namespace stockAgent
}  // namespace flamegpu
