#include "flamegpu/gpu/CUDASubAgent.h"


#include "flamegpu/gpu/CUDAScatter.h"
#include "flamegpu/gpu/CUDASubAgentStateList.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/model/AgentData.h"

CUDASubAgent::CUDASubAgent(const AgentData &description, const CUDAAgentModel& cuda_model, const std::unique_ptr<CUDAAgent> &_master_agent, const std::shared_ptr<SubAgentData> &_mapping)
    : CUDAAgent(description, cuda_model)
    , master_agent(_master_agent)
    , mapping(_mapping) {
    // Replace all mapped states in the statemap
    for (const auto &s_pair : mapping->states) {
        // Replace regular CUDAAgentStatelist with CUDASubAgentStateList
        state_map.at(s_pair.first) = std::make_shared<CUDASubAgentStateList>(*this, master_agent->getAgentStateList(s_pair.second), _mapping);
    }
    master_agent->setDependent(this);
}

void CUDASubAgent::resize(const unsigned int &newSize, const unsigned int &streamId) {
    const unsigned int old_size = getMaximumListSize();
    // Decide the new size, and resize master_agent (and mapped variables)
    master_agent->resize(newSize, streamId);
    if (old_size != getMaximumListSize()) {
        // Resize all unmapped states/variables
        for (auto &state : state_map) {
            state.second->resize(true);  // It auto pulls size from this->max_list_size
        }
    }
}
void CUDASubAgent::setPopulationData(const AgentPopulation& population) {
    // check that the gpu state lists have been initialised by a previous call to setInitialPopulationData
    assert(!state_map.empty());  // Check the removed code was redundant
    // If added back, this would require changing to selectively innit mapped items as CUDASubAgentStateList (see constructor)
    // if (state_map.empty()) {
    //     // create map of device state lists by traversing the state list
    //     for (const std::string &s : agent_description.states) {
    //         // allocate memory for each state list by creating a new Agent State List
    //         state_map.insert(CUDAStateMap::value_type(s, std::unique_ptr<CUDAAgentStateList>(new CUDAAgentStateList(*this))));
    //     }
    // }
    // Make sure population uses same agent description as was used to initialise the agent CUDASubAgent
    const std::string agent_name = agent_description.name;
    if (population.getAgentDescription() != agent_description) {
        THROW InvalidPopulationData("Error: Initial Population has a different agent description ('%s') "
            "to that which was used to initialise the CUDAAgent ('%s'). "
            "In CUDAAgent::setPopulationData()",
            population.getAgentName().c_str(),
            agent_description.name.c_str());
    }
    // check that the population maximums do not exceed the current maximum (as their will not be enough GPU memory to hold it)
    if (population.getMaximumStateListCapacity() > getMaximumListSize()) {
        // Resize the population exactly, setPopData is a whole population movement, no need for greedy resize
        // Unlikely to add new agents during simulation

        // Update capacity
        // Resize master_agent and mapped vars
        {
            // Get master agent desc
            const AgentData &master_desc = master_agent->getAgentDescription();
            for (auto &s : master_desc.states) {
                master_agent->getAgentStateList(s)->resize(false);
            }
        }  // Resize all unmapped states/variables
        for (auto &state : state_map) {
            state.second->resize(false);  // It auto pulls size from this->max_list_size
        }
    }
    /**set all population data to zero*/
    // This leaves master_agent vars which are not mapped unaffected
    // Particular use case is unclear here
    zeroAllStateVariableData();

    /**copy all population data to correct state map*/
    const std::set<std::string> &sm = agent_description.states;
    for (const std::string &s : sm) {
        // get an associated CUDA statemap pair
        CUDAStateMap::iterator i = state_map.find(s);

        /**check that the CUDAAgentStateList was found (should ALWAYS be the case)*/
        if (i == state_map.end()) {
            THROW InvalidMapEntry("Error: failed to find memory allocated for agent ('%s') state ('%s') "
                "In CUDAAgent::setPopulationData() ",
                "This should never happen!",
                population.getAgentName().c_str(), s.c_str());
        }
        // copy the data from the population state memory to the state_maps CUDAAgentStateList
        i->second->setAgentData(population.getReadOnlyStateMemory(i->first));
    }
}
unsigned int CUDASubAgent::getMaximumListSize() const {
    return master_agent->getMaximumListSize();
}
void CUDASubAgent::swapListsMaster(const std::string &master_state_source, const std::string &master_state_dest) {
    std::string src, dest;
    for (auto &m : mapping->states) {
        if (m.second == master_state_source)
            src = m.first;
        else if (m.second == master_state_dest)
            dest = m.first;
    }
    if (!src.empty() && !dest.empty()) {
        swap(state_map.at(src), state_map.at(dest));
        state_map.at(src)->swapDependants(state_map.at(dest));
        // Propagate
        if (dependent_agent) {
            dependent_agent->swapListsMaster(src, dest);
        }
    } else {
        assert(false);
        // Unclear how we should handle if an unmapped state is swapped into/out of
        // Potentially, the right course of action would be move it regardless, so it persists (idk how)/initialise a new one
    }
}
void CUDASubAgent::appendScatterMaps(
    const std::string &master_state_source,
    CUDAMemoryMap &merge_d_list,
    const std::string &master_state_dest,
    CUDAMemoryMap &merge_d_swap_list,
    VariableMap &merge_var_list,
    unsigned int streamId,
    unsigned int srcListSize,
    unsigned int destListSize) {
    // Map the master states
    std::string _src, _dest;
    for (auto &m : mapping->states) {
        if (m.second == master_state_source)
            _src = m.first;
        else if (m.second == master_state_dest)
            _dest = m.first;
    }
    // in[mapped]->out[mapped] = Mergelist is perfect match
    if (!_src.empty() && !_dest.empty()) {
        auto src_l = state_map.find(_src);
        auto dest_l = state_map.find(_dest);
        assert(srcListSize == src_l->second->getCUDAStateListSize());
        assert(destListSize == dest_l->second->getCUDAStateListSize());
        auto &src = src_l->second->getReadList();
        auto &dest = dest_l->second->getReadList();
        // Generate merge list
        const VariableMap &vars = agent_description.variables;
        // for each variable allocate a device array and add to map
        for (const auto &mm : vars) {
            // get the variable name
            const std::string sub_var_name = mm.first;
            const auto map = mapping->variables.find(sub_var_name);
            // Only handle variables which are not mapped
            if (map == mapping->variables.end()) {
                const std::string sub_var_merge_name = "_" + mm.first;  // Prepend reserved word _, to avoid clashes
                // Documentation isn't too clear, insert *should* throw an exception if we have a clash
                merge_d_list.insert({sub_var_merge_name, src.at(sub_var_name)});
                merge_d_swap_list.insert({sub_var_merge_name, dest.at(sub_var_name)});
                merge_var_list.insert({sub_var_merge_name, vars.at(sub_var_name)});
            }
        }
        // Propagate
        if (dependent_agent) {
            dependent_agent->appendScatterMaps(_src, merge_d_list, _dest, merge_d_swap_list, merge_var_list, streamId, srcListSize, destListSize);
        } else {
            // Perform scatter
            auto &cs = CUDAScatter::getInstance(streamId);
            cs.scatterAll(merge_var_list, merge_d_list, merge_d_swap_list, srcListSize, destListSize);
        }
        // After scatter has been performed, update list sizes
        dest_l->second->setCUDAStateListSize(destListSize + srcListSize);
        src_l->second->setCUDAStateListSize(0);
    } else if (_src.empty() && _dest.empty()) {
        // Scatter only, can't propogate if cant map both states
        auto &cs = CUDAScatter::getInstance(streamId);
        cs.scatterAll(merge_var_list, merge_d_list, merge_d_swap_list, srcListSize, destListSize);
    } else {
        // in[mapped]->out[not mapped] = ?????????
        // in[not mapped]->out[mapped] = ????????
        assert(false);
        // Unclear how we should handle if an unmapped state is swapped into/out of
        // Potentially, the right course of action would be move it regardless, so it persists (idk how)/initialise a new one
    }
}
