#include "flamegpu/sim/RunPlan.h"
#include "flamegpu/sim/RunPlanVec.h"

#include "flamegpu/model/ModelDescription.h"

RunPlan::RunPlan(const ModelDescription &model)
    : RunPlan(std::make_shared<std::unordered_map<std::string, EnvironmentDescription::PropData> const>(model.model->environment->getPropertiesMap()),
      model.model->exitConditions.size() + model.model->exitConditionCallbacks.size() > 0) { }
RunPlan::RunPlan(const std::shared_ptr<const std::unordered_map<std::string, EnvironmentDescription::PropData>>  &environment, const bool &allow_0)
    : random_seed(0)
    , steps(0)
    , environment(environment)
    , allow_0_steps(allow_0) { }

RunPlan& RunPlan::operator=(const RunPlan& other) {
    this->random_seed = other.random_seed;
    this->steps = other.steps;
    this->environment = other.environment;
    this->allow_0_steps = other.allow_0_steps;
    this->output_subdirectory = other.output_subdirectory;
    this->allow_0_steps = other.allow_0_steps;
    for (auto &i : other.property_overrides)
        this->property_overrides.emplace(i.first, Any(i.second));
    return *this;
}
void RunPlan::setRandomSimulationSeed(const unsigned int &_random_seed) { random_seed = _random_seed; }
void RunPlan::setSteps(const unsigned int &_steps) {
    if (_steps == 0 && !allow_0_steps) {
        throw std::out_of_range("Model description requires atleast 1 exit condition to have unlimited steps, "
            "in RunPlan::setSteps()");
    }
    steps = _steps;
}
void RunPlan::setOutputSubdirectory(const std::string &subdir) {
    output_subdirectory = subdir;
}

unsigned int RunPlan::getRandomSimulationSeed() const {
    return random_seed;
}
unsigned int RunPlan::getSteps() const {
    return steps;
}
std::string RunPlan::getOutputSubdirectory() const {
    return output_subdirectory;
}

RunPlanVec RunPlan::operator+(const RunPlan& rhs) const {
    // Validation
    if (*rhs.environment != *this->environment) {
        THROW InvalidArgument("RunPlan is for a different ModelDescription, "
            "in ::operator+(RunPlan, RunPlan)");
    }
    // Operation
    RunPlanVec rtn(this->environment, this->allow_0_steps);
    rtn+=*this;
    rtn+=rhs;
    return rtn;
}
RunPlanVec RunPlan::operator+(const RunPlanVec& rhs) const {
    // This function is defined internally inside both RunPlan and RunPlanVec as it's the only way to both pass CI and have SWIG build
    // Validation
    if (*rhs.environment != *this->environment) {
        THROW InvalidArgument("RunPlan is for a different ModelDescription, "
            "in ::operator+(RunPlan, RunPlanVec)");
    }
    // Operation
    RunPlanVec rtn(rhs);
    rtn+=*this;
    return rtn;
}
RunPlanVec RunPlan::operator*(const unsigned int& rhs) const {
    // Operation
    RunPlanVec rtn(this->environment, this->allow_0_steps);
    for (unsigned int i = 0; i < rhs; ++i) {
        rtn+=*this;
    }
    return rtn;
}
