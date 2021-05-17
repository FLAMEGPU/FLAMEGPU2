#include "flamegpu/sim/RunPlanVec.h"
#include "flamegpu/model/ModelDescription.h"

namespace flamegpu {

RunPlanVec::RunPlanVec(const ModelDescription &model, unsigned int initial_length)
    : std::vector<RunPlan>(initial_length, RunPlan(model)),
      rand(std::random_device()())
    , environment(std::make_shared<std::unordered_map<std::string, EnvironmentDescription::PropData> const>(model.model->environment->getPropertiesMap()))
    , allow_0_steps(model.model->exitConditions.size() + model.model->exitConditionCallbacks.size() > 0) {
    this->resize(initial_length, RunPlan(environment, allow_0_steps));
}

RunPlanVec::RunPlanVec(const std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>> &_environment, const bool &_allow_0_steps)
    : std::vector<RunPlan>(),
      rand(std::random_device()())
    , environment(_environment)
    , allow_0_steps(_allow_0_steps) { }
void RunPlanVec::setRandomSimulationSeed(const unsigned int &initial_seed, const unsigned int &step) {
    unsigned int current_seed = initial_seed;
    for (auto &i : *this) {
        i.setRandomSimulationSeed(current_seed);
        current_seed += step;
    }
}
void RunPlanVec::setSteps(const unsigned int &steps) {
    if (steps == 0 && !allow_0_steps) {
        throw std::out_of_range("Model description requires atleast 1 exit condition to have unlimited steps, "
            "in RunPlanVec::setSteps()");
    }
    for (auto &i : *this) {
        i.setSteps(steps);
    }
}
void RunPlanVec::setOutputSubdirectory(const std::string &subdir) {
    for (auto &i : *this) {
        i.setOutputSubdirectory(subdir);
    }
}
void RunPlanVec::setRandomPropertySeed(const unsigned int &seed) {
    rand.seed(seed);
}

RunPlanVec RunPlanVec::operator+(const RunPlan& rhs) const {
    // This function is defined internally inside both RunPlan and RunPlanVec as it's the only way to both pass CI and have SWIG build
    // Validation
    if (*rhs.environment != *this->environment) {
        THROW InvalidArgument("RunPlan is for a different ModelDescription, "
            "in ::operator+(RunPlanVec, RunPlan)");
    }
    // Operation
    RunPlanVec rtn(*this);
    rtn+=rhs;
    return rtn;
}
RunPlanVec RunPlanVec::operator+(const RunPlanVec& rhs) const {
    // Validation
    if (*rhs.environment != *this->environment) {
        THROW InvalidArgument("RunPlanVecs are for different ModelDescriptions, "
            "in ::operator+(RunPlanVec, RunPlanVec)");
    }
    // Operation
    RunPlanVec rtn(*this);
    rtn+=rhs;
    return rtn;
}
RunPlanVec& RunPlanVec::operator+=(const RunPlan& rhs) {
    // Validation
    if (*rhs.environment != *this->environment) {
        THROW InvalidArgument("RunPlan is for a different ModelDescription, "
            "in ::operator+=(RunPlanVec, RunPlan)");
    }
    // Update shared_ptr to env
    RunPlan rhs_copy = rhs;
    rhs_copy.environment = environment;
    // Operation
    this->push_back(rhs_copy);
    return *this;
}
RunPlanVec& RunPlanVec::operator+=(const RunPlanVec& rhs) {
    // Validation
    if (this == &rhs) {
        return *this*=2;
    }
    if (*rhs.environment != *this->environment) {
        THROW InvalidArgument("RunPlan is for a different ModelDescription, "
            "in ::operator+=(RunPlanVec, RunPlan)");
    }
    // Operation
    this->reserve(size() + rhs.size());
    // Iterate, because insert would require RunPlan::operator==
    for (const auto &i : rhs) {
        // Update shared_ptr to env
        RunPlan i_copy = i;
        i_copy.environment = environment;
        this->push_back(i);
    }
    return *this;
}
RunPlanVec& RunPlanVec::operator*=(const unsigned int& rhs) {
    RunPlanVec copy(*this);
    this->clear();
    this->reserve(copy.size() * rhs);
    for (unsigned int i = 0; i < rhs; ++i) {
        // Iterate, because insert would require RunPlan::operator==
        for (const auto &j : copy) {
            this->push_back(j);
        }
    }
    return *this;
}
RunPlanVec RunPlanVec::operator*(const unsigned int& rhs) const {
    RunPlanVec rtn(this->environment, this->allow_0_steps);
    rtn.reserve(size() * rhs);
    for (unsigned int i = 0; i < rhs; ++i) {
        // Iterate, because insert would require RunPlan::operator==
        for (const auto &j : *this) {
            rtn.push_back(j);
        }
    }
    return *this;
}

}  // namespace flamegpu
