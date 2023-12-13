#include <chrono>
#include <thread>

#ifdef FLAMEGPU_ENABLE_MPI
#include <mpi.h>
#endif

#include "flamegpu/flamegpu.h"
#include "flamegpu/simulation/detail/MPIEnsemble.h"
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
 * Error tests do not currently control whether the error occurs on rank==0 or rank != 0
 * These errors are handled differently, so it would benefit from duplicate tests (for multi-rank runs)
 *
 * MPI_Init() and MPI_Finalize() can only be called once each
 *
 * Ideally these tests should be executed with less mpi ranks than gpus, the same number of mpi ranks as gpus, and more mpi ranks than gpus.
 */
#ifdef FLAMEGPU_ENABLE_MPI
int has_error = 0;
FLAMEGPU_STEP_FUNCTION(model_step) {
    int counter = FLAMEGPU->environment.getProperty<int>("counter");
    counter+=1;
    FLAMEGPU->environment.setProperty<int>("counter", counter);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
FLAMEGPU_STEP_FUNCTION(throw_exception_rank_0) {
    int world_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 0) return;
    const int counter = FLAMEGPU->environment.getProperty<int>("counter");
    const int init_counter = FLAMEGPU->environment.getProperty<int>("init_counter");
    if (FLAMEGPU->getStepCounter() == 1 && counter > 6 && has_error < 2) {
        ++has_error;
        // printf("exception counter: %u, init_counter: %u\n", counter, init_counter);
        throw std::runtime_error("Exception thrown by host fn throw_exception()");
    }
}
FLAMEGPU_STEP_FUNCTION(throw_exception_rank_1) {
    int world_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank != 1) return;
    const int counter = FLAMEGPU->environment.getProperty<int>("counter");
    const int init_counter = FLAMEGPU->environment.getProperty<int>("init_counter");
    if (FLAMEGPU->getStepCounter() == 1 && counter > 6 && has_error < 2) {
        ++has_error;
        // printf("exception counter: %u, init_counter: %u\n", counter, init_counter);
        throw std::runtime_error("Exception thrown by host fn throw_exception()");
    }
}
class TestMPIEnsemble : public testing::Test {
 protected:
    void SetUp() override {
        has_error = 0;
        int flag = 0;
        MPI_Finalized(&flag);
        if (flag) {
            GTEST_SKIP() << "Skipping MPI test, MPI_Finalize() has been called";
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
        model->Environment().newProperty<int>("init_counter", -1);
        model->newAgent("agent");
        model->newLayer().addHostFunction(model_step);
    }
    void initPlans() {
        // 10 runs per rank, steps which take >100ms each,
        // therefore each sim will take >1 second.
        // Estimate 10 seconds for full test across any world size
        const int RUNS_PER_RANK = 10;
        plans = new flamegpu::RunPlanVector(*model, RUNS_PER_RANK * world_size);
        plans->setSteps(10);
        plans->setPropertyLerpRange<int>("counter", 0, (RUNS_PER_RANK * world_size) - 1);
        plans->setPropertyLerpRange<int>("init_counter", 0, (RUNS_PER_RANK * world_size) - 1);
    }
    void initExitLoggingConfig() {
        exit_log_cfg = new flamegpu::LoggingConfig(*model);
        exit_log_cfg->logEnvironment("counter");
    }
    void initEnsemble() {
        initPlans();
        initExitLoggingConfig();
        ensemble = new flamegpu::CUDAEnsemble(*model);
        ensemble->Config().concurrent_runs = 1;
        ensemble->setExitLog(*exit_log_cfg);
    }
    void validateLogs() {
        // Validate results
        // @note Best we can currently do is check logs of each runner have correct results
        // @note Ideally we'd validate between nodes to ensure all runs have been completed
        const std::map<unsigned int, RunLog> logs = ensemble->getLogs();
        for (const auto &[index, log] : logs) {
            const ExitLogFrame& exit_log = log.getExitLog();
            // Get a logged environment property
            const int counter = exit_log.getEnvironmentProperty<int>("counter");
            EXPECT_EQ(counter, index + 10);
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
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const unsigned int err_count = ensemble->simulate(*plans);
    EXPECT_EQ(err_count, 0u);
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(errors.empty() || errors.find("Warning: MPI Ensemble launched with") != std::string::npos);  // can't fully match the error message as it contains mpirun/machine specific numbers
    } else {
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    validateLogs();
}
TEST_F(TestMPIEnsemble, success_verbose) {
    initEnsemble();
    ensemble->Config().verbosity = Verbosity::Verbose;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    const unsigned int err_count = ensemble->simulate(*plans);
    EXPECT_EQ(err_count, 0u);
    // Get stderr and stdout
    std::string output = testing::internal::GetCapturedStdout();
    std::string errors = testing::internal::GetCapturedStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") != std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty() || errors.find("Warning: MPI Ensemble launched with") != std::string::npos);  // can't fully match the error message as it contains mpirun/machine specific numbers
    } else {
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    validateLogs();
}

TEST_F(TestMPIEnsemble, error_off_rank_0) {
    model->newLayer().addHostFunction(throw_exception_rank_0);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Off;
    unsigned int err_count = 0;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(err_count = ensemble->simulate(*plans));
    // Get stderr and stdout
    const std::string output = testing::internal::GetCapturedStdout();
    const std::string errors = testing::internal::GetCapturedStderr();
    // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
    // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        // With error off, we would expect to see run index 10 fail
        // Therefore 1 returned instead of 0
        EXPECT_EQ(err_count, 2u);
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(errors.find("Warning: Run ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
        EXPECT_TRUE(errors.find(" failed on rank ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
    } else {
        // Capture stderr and stdout
        // Only rank 0 returns the error count
        EXPECT_EQ(err_count, 0u);
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_slow_rank_0) {
    model->newLayer().addHostFunction(throw_exception_rank_0);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Slow;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_THROW(ensemble->simulate(*plans), flamegpu::exception::EnsembleError);
        // @todo can't capture total number of successful/failed runs
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(errors.find("Warning: Run ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
        EXPECT_TRUE(errors.find(" failed on rank ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
    } else {
        // Only rank 0 raises exception
        EXPECT_NO_THROW(ensemble->simulate(*plans));
        // @todo can't capture total number of successful/failed runs
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }
    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_fast_rank_0) {
    model->newLayer().addHostFunction(throw_exception_rank_0);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Fast;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_THROW(ensemble->simulate(*plans), flamegpu::exception::EnsembleError);
        // @todo can't capture total number of successful/failed runs
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
#ifdef _DEBUG
        EXPECT_TRUE(errors.find("Run ") != std::string::npos);  // E.g. Run 5 failed on rank 0, device 1, thread 0 with exception:
        EXPECT_TRUE(errors.find(" failed on rank ") != std::string::npos);  // E.g. Run 5 failed on rank 0, device 1, thread 0 with exception:
#else
        EXPECT_TRUE(errors.empty() || errors.find("Warning: MPI Ensemble launched with") != std::string::npos);  // can't fully match the error message as it contains mpirun/machine specific numbers
#endif
    } else {
        // Only rank 0 raises exception and captures total number of successful/failed runs
        EXPECT_NO_THROW(ensemble->simulate(*plans));
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_off_rank_1) {
    if (world_size == 1) {
        GTEST_SKIP() << "world_size==1, test not applicable.";
    }
    model->newLayer().addHostFunction(throw_exception_rank_1);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Off;
    unsigned int err_count = 0;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(err_count = ensemble->simulate(*plans));
    // Get stderr and stdout
    const std::string output = testing::internal::GetCapturedStdout();
    const std::string errors = testing::internal::GetCapturedStderr();
    // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
    // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        // With error off, we would expect to see run index 10 fail
        // Therefore 1 returned instead of 0
        EXPECT_EQ(err_count, 2u);
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(errors.find("Warning: Run ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
        EXPECT_TRUE(errors.find(" failed on rank ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
    } else {
        // Capture stderr and stdout
        // Only rank 0 returns the error count
        EXPECT_EQ(err_count, 0u);
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }
    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_slow_rank_1) {
    if (world_size == 1) {
        GTEST_SKIP() << "world_size==1, test not applicable.";
    }
    model->newLayer().addHostFunction(throw_exception_rank_1);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Slow;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_THROW(ensemble->simulate(*plans), flamegpu::exception::EnsembleError);
        // @todo can't capture total number of successful/failed runs
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(errors.find("Warning: Run ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
        EXPECT_TRUE(errors.find(" failed on rank ") != std::string::npos);  // E.g. Warning: Run 10/10 failed on rank 0, device 0, thread 0 with exception:
    } else {
        // Only rank 0 raises exception
        EXPECT_NO_THROW(ensemble->simulate(*plans));
        // @todo can't capture total number of successful/failed runs
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }
    // Existing logs should still validate
    validateLogs();
}
TEST_F(TestMPIEnsemble, error_fast_rank_1) {
    if (world_size == 1) {
        GTEST_SKIP() << "world_size==1, test not applicable.";
    }
    model->newLayer().addHostFunction(throw_exception_rank_1);
    initEnsemble();
    ensemble->Config().error_level = CUDAEnsemble::EnsembleConfig::Fast;
    // Capture stderr and stdout
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    // Expect no warnings (stderr) but outputs on progress and timing
    if (world_rank == 0) {
        EXPECT_THROW(ensemble->simulate(*plans), flamegpu::exception::EnsembleError);
        // @todo can't capture total number of successful/failed runs
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.find("MPI ensemble assigned run ") != std::string::npos);   // E.g. MPI ensemble assigned run %d/%u to rank %d
#ifdef _DEBUG
        EXPECT_TRUE(errors.find("Run ") != std::string::npos);  // E.g. Run 5 failed on rank 1, device 1, thread 0 with exception:
        EXPECT_TRUE(errors.find(" failed on rank ") != std::string::npos);  // E.g. Run 5 failed on rank 1, device 1, thread 0 with exception:
#else
        EXPECT_TRUE(errors.empty() || errors.find("Warning: MPI Ensemble launched with") != std::string::npos);  // can't fully match the error message as it contains mpirun/machine specific numbers
#endif
    } else {
        // Only rank 0 raises exception and captures total number of successful/failed runs
        EXPECT_NO_THROW(ensemble->simulate(*plans));
        // Get stderr and stdout
        const std::string output = testing::internal::GetCapturedStdout();
        const std::string errors = testing::internal::GetCapturedStderr();
        // printf("[%d]output:\n:%s\n", world_rank, output.c_str());
        // printf("[%d]errors:\n:%s\n", world_rank, errors.c_str());
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }

    // Existing logs should still validate
    validateLogs();
}
// This test doesn't want to use the fixture, so must use a differnet test suite name
TEST(TestMPIEnsembleNoFixture, devicesForThisRank) {
    // Call the static, testable version of devicesForThisRank assigns the correct number of devices to the "current" rank, faking the mpi rank and size to make this testable regardless of mpirun config.

    // with 1 local mpi rank, the full set of devices should be used regarldes of count
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0}, 1, 0), std::set<int>({0}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1}, 1, 0), std::set<int>({0, 1}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({1, 2, 3}, 1, 0), std::set<int>({1, 2, 3}));

    // with the same number of ranks as gpus, each rank should get thier respective gpu
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 4, 0), std::set<int>({0}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 4, 1), std::set<int>({1}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 4, 2), std::set<int>({2}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 4, 3), std::set<int>({3}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({2, 3}, 2, 0), std::set<int>({2}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({2, 3}, 2, 1), std::set<int>({3}));

    // With fewer ranks than gpus, gpus should be load balanced, with lower ranks having extra gpus if the number is not divisible
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 2, 0), std::set<int>({0, 1}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 2, 1), std::set<int>({2, 3}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2}, 2, 0), std::set<int>({0, 1}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2}, 2, 1), std::set<int>({2}));

    // with more ranks than gpus, each rank should get 1 or 0 gpus.
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0}, 3, 0), std::set<int>({0}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0}, 3, 1), std::set<int>({}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0}, 3, 2), std::set<int>({}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 6, 0), std::set<int>({0}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 6, 1), std::set<int>({1}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 6, 2), std::set<int>({2}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 6, 3), std::set<int>({3}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 6, 4), std::set<int>({}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({0, 1, 2, 3}, 6, 5), std::set<int>({}));

    // std::set is sorted, not ordered, so the selected device will use lower device ids for lower ranks
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({3, 2, 1}, 3, 0), std::set<int>({1}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({3, 2, 1}, 3, 1), std::set<int>({2}));
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({3, 2, 1}, 3, 2), std::set<int>({3}));

    // no devices, set should be empty regardless of mpi size / rank
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({}, 1, 0), std::set<int>());
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({}, 2, 0), std::set<int>());
    EXPECT_EQ(flamegpu::detail::MPIEnsemble::devicesForThisRank({}, 2, 1), std::set<int>());
}
#else
TEST(TestMPIEnsemble, DISABLED_success) { }
TEST(TestMPIEnsemble, DISABLED_success_verbose) { }
TEST(TestMPIEnsemble, DISABLED_error_off_rank_0) { }
TEST(TestMPIEnsemble, DISABLED_error_slow_rank_0) { }
TEST(TestMPIEnsemble, DISABLED_error_fast_rank_0) { }
TEST(TestMPIEnsemble, DISABLED_error_off_rank_1) { }
TEST(TestMPIEnsemble, DISABLED_error_slow_rank_1) { }
TEST(TestMPIEnsemble, DISABLED_error_fast_rank_1) { }
TEST(TestMPIEnsembleNoFixture, DISABLED_devicesForThisRank) { }
#endif

}  // namespace test_mpi_ensemble
}  // namespace tests
}  // namespace flamegpu
