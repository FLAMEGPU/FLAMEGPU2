#ifndef INCLUDE_FLAMEGPU_SIM_SIMLOGGER_H_
#define INCLUDE_FLAMEGPU_SIM_SIMLOGGER_H_

#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <string>
#include <condition_variable>

#include "flamegpu/sim/LogFrame.h"

class RunPlanVec;

class SimLogger {
    friend class CUDAEnsemble;
    SimLogger(const std::vector<RunLog> &run_logs,
        const RunPlanVec &run_plans,
        const std::string &out_directory,
        const std::string &out_format,
        std::queue<unsigned int> &log_export_queue,
        std::mutex &log_export_queue_mutex,
        std::condition_variable &log_export_queue_cdn);

    std::thread thread;
    void start();
    // External references
    const std::vector<RunLog> &run_logs;
    const RunPlanVec &run_plans;
    const std::string &out_directory;
    const std::string &out_format;
    std::queue<unsigned int> &log_export_queue;
    std::mutex &log_export_queue_mutex;
    std::condition_variable &log_export_queue_cdn;
};

#endif  // INCLUDE_FLAMEGPU_SIM_SIMLOGGER_H_
