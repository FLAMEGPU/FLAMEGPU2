#include <chrono>
#include <thread>

#ifdef FLAMEGPU_ENABLE_MPI
#include <mpi.h>
#endif

#include "flamegpu/flamegpu.h"
#include "gtest/gtest.h"


namespace flamegpu {
namespace tests {
namespace test_mpi_ensemble {
/**
 * Testing MPI with GoogleTest is somewhat impractical
 * Therefore this suite intends to run a single complete test depending on the run state
 * Not built with FLAMEGPU_ENABLE_MPI: All TestMPIEnsemble disabled
 * Not executed with mpirun: MPI initialises with a world_size 1, the model should execute as normal
 * Executed on a single node TestMPIEnsemble 'mpirun -n 4 ./bin/Debug/tests', TestMPIEnsemble.local will run
 * Executed on multiple nodes TestMPIEnsemble 'mpirun -n $SLURM_JOB_NUM_NODES -hosts $SLURM_JOB_NODELIST ./bin/Debug/tests', TestMPIEnsemble.multi will run
 *
 * MPI_Init() and MPI_Finalize() can only be called once each
 */
#ifdef FLAMEGPU_ENABLE_MPI
FLAMEGPU_STEP_FUNCTION(model_step) {
    int counter = FLAMEGPU->environment.getProperty<int>("counter");
    counter+=1;
    FLAMEGPU->environment.setProperty<int>("counter", counter);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
class TestMPIEnsemble : public testing::Test {
 protected:
    void SetUp() override {
        int flag = 0;
        MPI_Finalized(&flag);
        if (flag) {
            // Multi node MPI run
            GTEST_SKIP() << "Skipping single-node MPI test, MPI_Finalize() has been called";
            return;
        }
        // Init MPI, fetch rank and size
        int thread_provided = 0;
        // MPI can only be init once, for certain test cases we do some initial MPI comms for setup
        MPI_Initialized(&flag);
        if (!flag) {
            // MPI single means that only the main thread will perform MPI actions
            MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &thread_provided);
            if (thread_provided != MPI_THREAD_SINGLE) {
                THROW exception::UnknownInternalError("MPI unable to provide MPI_THREAD_SINGLE support");
            }
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        // Create a shared memory split
        MPI_Comm group;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &group);
        MPI_Comm_size(group, &group_size);
        if (group_size != world_size) {
            // Multi node MPI run
            GTEST_SKIP() << "Skipping single-node MPI test, world size exceeds shared memory split size.";
            return;
        }
        initModel();
        initPlans();
        initExitLoggingConfig();
    }
    void TearDown() override {
        if (exit_log_cfg) delete exit_log_cfg;
        if (plans) delete plans;
        if (model) delete model;
    }
    void initModel() {
        model = new flamegpu::ModelDescription("MPITest");
        model->Environment().newProperty<int>("counter", -1);
        model->Environment().newProperty<int>("counter_init", -1);
        model->newAgent("agent");
        model->newLayer().addHostFunction(model_step);
    }
    void initPlans() {
        // 100 runs, each 10 steps which take >100ms each,
        // therefore each sim will take >1 second.
        // Estimate 30 seconds for full test across 4 threads
        plans = new flamegpu::RunPlanVector(*model, 100);
        plans->setSteps(10);
        plans->setPropertyLerpRange<int>("counter", 0, 99);
        plans->setPropertyLerpRange<int>("counter_init", 0, 99);
    }
    void initExitLoggingConfig() {
        exit_log_cfg = new flamegpu::LoggingConfig(*model);
        exit_log_cfg->logEnvironment("counter");
        exit_log_cfg->logEnvironment("counter_init");
    }
    int world_rank = -1;
    int group_size = -1;
    int world_size = -1;
    flamegpu::ModelDescription *model = nullptr;
    flamegpu::RunPlanVector *plans = nullptr;
    flamegpu::LoggingConfig *exit_log_cfg = nullptr;
};
TEST_F(TestMPIEnsemble, local) {
    if (world_size < 1) return;
    if (group_size != world_size) {
        // Multi node MPI run
        GTEST_SKIP() << "Skipping single-node MPI test, world size exceeds shared memory split size.";
        return;
    }
    // Single node MPI run
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < world_size) {
        MPI_Finalize();
        GTEST_SKIP() << "Skipping single-node MPI test, world size (" << world_size << ") exceeds GPU count (" << deviceCount << "), this would cause test to stall.";
    }
    flamegpu::CUDAEnsemble ensemble(*model);
    ensemble.Config().concurrent_runs = 1;
    if (world_size > 1) {
        ensemble.Config().devices = {world_rank};
    }
    ensemble.setExitLog(*exit_log_cfg);

    ensemble.simulate(*plans);

    // Validate results
    // @note Best we can currently do is check logs of each runner have correct results
    // @note Ideally we'd validate between nodes to ensure all runs have been completed
    const std::vector<RunLog> logs = ensemble.getLogs();
    for (const auto &log : logs) {
        const ExitLogFrame& exit_log = log.getExitLog();
        // EXPECT_EQ(exit_log.getStepCount(), 10);
        if (exit_log.getStepCount()) {  // Temp, currently every runner gets all logs but unhandled ones are empty
            // Get a logged environment property
            const int counter = exit_log.getEnvironmentProperty<int>("counter");
            const int counter_init = exit_log.getEnvironmentProperty<int>("counter_init");  // @todo Can't access run index via RunLog
            EXPECT_EQ(counter, counter_init + 10);
        }
    }
}
TEST_F(TestMPIEnsemble, multi) {
    if (world_size < 1) return;
    if (group_size == world_size) {
        // Single node MPI run
        GTEST_SKIP() << "Skipping multi-node MPI test, world size equals shared memory split size.";
        return;
    }
    // Multi node MPI run
    if (world_size < 1) return;
    if (group_size == world_size) {
        // Single node MPI run
        GTEST_SKIP() << "Skipping multi-node MPI test, world size equals shared memory split size.";
        return;
    }
    // Multi node MPI run
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < group_size) {
        MPI_Finalize();
        GTEST_SKIP() << "Skipping multi-node MPI test on this node, group size (" << group_size << ") exceeds GPU count (" << deviceCount << "), this would cause test to stall.";
    }
    flamegpu::CUDAEnsemble ensemble(*model);
    ensemble.Config().concurrent_runs = 1;
    if (group_size > 1) {
        ensemble.Config().devices = {world_rank};
    }
    ensemble.setExitLog(*exit_log_cfg);

    ensemble.simulate(*plans);

    // Validate results
    // @note Best we can currently do is check logs of each runner have correct results
    // @note Ideally we'd validate between nodes to ensure all runs have been completed
    const std::vector<RunLog> logs = ensemble.getLogs();
    for (const auto &log : logs) {
        const ExitLogFrame& exit_log = log.getExitLog();
        // EXPECT_EQ(exit_log.getStepCount(), 10);
        if (exit_log.getStepCount()) {  // Temp, currently every runner gets all logs but unhandled ones are empty
            // Get a logged environment property
            const int counter = exit_log.getEnvironmentProperty<int>("counter");
            const int counter_init = exit_log.getEnvironmentProperty<int>("counter_init");  // @todo Can't access run index via RunLog
            EXPECT_EQ(counter, counter_init + 10);
        }
    }
}
#else
TEST(TestMPIEnsemble, DISABLED_local_test) { }
TEST(TestMPIEnsemble, DISABLED_multi_test) { }
#endif

}  // namespace test_mpi_ensemble
}  // namespace tests
}  // namespace flamegpu
