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
FLAMEGPU_STEP_FUNCTION(throw_exception) {
    const int counter = FLAMEGPU->environment.getProperty<int>("counter");
    if (FLAMEGPU->getStepCounter() == 1 && counter == 12) {
        throw flamegpu::exception::VersionMismatch("Counter - %d", counter);
    }
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
        if (world_size < 1) {
            GTEST_SKIP() << "world_size<1, something went wrong.";
        }
        initModel();
        initPlans();
        initExitLoggingConfig();
    }
    void TearDown() override {
        if (ensemble) delete ensemble;
        if (exit_log_cfg) delete exit_log_cfg;
        if (plans) delete plans;
        if (model) delete model;
    }
    void initModel() {
        model = new flamegpu::ModelDescription("MPITest");
        model->Environment().newProperty<int>("counter", -1);
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
    }
    void initExitLoggingConfig() {
        exit_log_cfg = new flamegpu::LoggingConfig(*model);
        exit_log_cfg->logEnvironment("counter");
    }
    void initEnsemble() {
        ensemble = new flamegpu::CUDAEnsemble (*model);
        ensemble->Config().concurrent_runs = 1;
        if (group_size == world_size) {
            // Single node MPI run
            int deviceCount = 0;
            gpuErrchk(cudaGetDeviceCount(&deviceCount));
            if (deviceCount < world_size) {
                MPI_Finalize();
                GTEST_SKIP() << "Skipping single-node MPI test, world size (" << world_size << ") exceeds GPU count (" << deviceCount << "), this would cause test to stall.";
            }
            ensemble->Config().concurrent_runs = 1;
            if (world_size > 1) {
                ensemble->Config().devices = {world_rank};
            }
        }
        ensemble->setExitLog(*exit_log_cfg);
    }
    void validateLogs() {
        // Validate results
        // @note Best we can currently do is check logs of each runner have correct results
        // @note Ideally we'd validate between nodes to ensure all runs have been completed
        const std::map<unsigned int, RunLog> logs = ensemble.getLogs();
        for (const auto &[index, log] : logs) {
            const ExitLogFrame& exit_log = log.getExitLog();
            // EXPECT_EQ(exit_log.getStepCount(), 10);
            if (exit_log.getStepCount()) {  // Temp, currently every runner gets all logs but unhandled ones are empty
                // Get a logged environment property
                const int counter = exit_log.getEnvironmentProperty<int>("counter");
                EXPECT_EQ(counter, index + 10);
            }
        }
    }
    int world_rank = -1;
    int group_size = -1;
    int world_size = -1;
    flamegpu::ModelDescription *model = nullptr;
    flamegpu::RunPlanVector *plans = nullptr;
    flamegpu::LoggingConfig *exit_log_cfg = nullptr;
    flamegpu::CUDAEnsemble *ensemble = nullptr;
};
TEST_F(TestMPIEnsemble, success) {
    initEnsemble();
    const unsigned int successful_runs = ensemble->simulate(*plans);
    EXPECT_EQ(successful_runs, 100u);
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_TRUE(output.find("CUDAEnsemble progress") != std::string::npos);   // E.g. CUDAEnsemble progress: 1/2
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") != std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty());
    } else {
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    validateLogs();
}
TEST_F(TestMPIEnsemble, error_off) {
    model->newLayer().addHostFunction(throw_exception);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Off;
    const unsigned int successful_runs = 0;
    EXPECT_NO_THROW(successful_runs = ensemble->simulate(*plans));
    // With error off, we would expect to see run index 12 fail
    // Therefore 99 returned instead of 100
    EXPECT_EQ(successful_runs, 99u);
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_TRUE(output.find("CUDAEnsemble progress") != std::string::npos);   // E.g. CUDAEnsemble progress: 1/2
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") != std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty());
    } else {
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_slow) {
    model->newLayer().addHostFunction(throw_exception);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Slow;
    EXPECT_THROW(ensemble->simulate(*plans), flamegpu::exception::EnsembleError);
    // @todo can't capture total number of successful/failed runs
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_TRUE(output.find("CUDAEnsemble progress") != std::string::npos);   // E.g. CUDAEnsemble progress: 1/2
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") != std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty());
    } else {
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_fast) {
    model->newLayer().addHostFunction(throw_exception);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Fast;
    EXPECT_THROW(ensemble->simulate(*plans), flamegpu::exception::EnsembleError);
    // @todo can't capture total number of successful/failed runs
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_TRUE(output.find("CUDAEnsemble progress") != std::string::npos);   // E.g. CUDAEnsemble progress: 1/2
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") != std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty());
    } else {
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    // Existing logs should still validate
    validateLogs();
}
#else
TEST(TestMPIEnsemble, DISABLED_success) { }
TEST(TestMPIEnsemble, DISABLED_error_off) { }
TEST(TestMPIEnsemble, DISABLED_error_slow) { }
TEST(TestMPIEnsemble, DISABLED_error_fast) { }
#endif

}  // namespace test_mpi_ensemble
}  // namespace tests
}  // namespace flamegpu
