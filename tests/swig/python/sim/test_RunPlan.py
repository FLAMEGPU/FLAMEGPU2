import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint

# Test the RunPlan interface
class TestRunPlan(TestCase):
    def test_constructor(self):
        # Create a model 
        model = pyflamegpu.ModelDescription("test")
        plan = None
        plan = pyflamegpu.RunPlan(model)
        assert plan != None
        plan = None

    def test_setRandomSimulationSeed(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Get the original simulation seed. Cannot compare for an expected value as any uint64_t is potentially legitimate.
        plan.getRandomSimulationSeed()
        # Set to max 32 bit value +1
        uint32_max = 2 ** 32 - 1
        newSimulationSeed = uint32_max + 1
        plan.setRandomSimulationSeed(newSimulationSeed)
        # Get the value again, and expect it to be the set value. It is not guaranteed to not be the original random value.
        simulationSeedUpdated = plan.getRandomSimulationSeed()
        assert newSimulationSeed == simulationSeedUpdated
        # Set it again, this time passing a narrower number.
        # @todo - ensure this is actually using a 32 bit integer if possible?
        narrowSimulationSeed = 12
        plan.setRandomSimulationSeed(narrowSimulationSeed)
        # Get the seed again, into a narrow value
        narrowSimulationSeedUpdated = plan.getRandomSimulationSeed()
        assert narrowSimulationSeed == narrowSimulationSeedUpdated

    def test_setSteps(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Get the default value
        steps = plan.getSteps()
        assert steps == 1
        # Set a new value
        newSteps = 12
        plan.setSteps(newSteps)
        # Get the updated value and compare
        updatedSteps = plan.getSteps()
        assert updatedSteps == newSteps

        # Expected exception tests
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setSteps(0)
        assert e.value.type() == "OutOfBoundsException"

    def test_setOutputSubdirectory(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Set the subdirectory to a non empty string
        newSubdir = "test"
        plan.setOutputSubdirectory(newSubdir)
        # Get the original value
        updatedSubdir = plan.getOutputSubdirectory()
        # By default this is an empty string
        assert updatedSubdir == newSubdir

    def test_setProperty(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Add some properties to the model, using a range of types.
        environment = model.Environment()
        environment.newPropertyFloat("f", 1.0)
        environment.newPropertyInt("i", -1)
        environment.newPropertyUInt("u", 1)
        environment.newPropertyArrayFloat("f_a", 3, (-1.0, 0.0, 1.0))
        environment.newPropertyArrayInt("i_a", 3, (-1, 0, 1))
        environment.newPropertyArrayUInt("u_a", 3, (0, 1, 2))
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Set properties to new values
        # Compare the old and new values, to ensure that thy do not match
        # RunPlan::setProperty(const std::string &name, const T&value)
        plan.setPropertyFloat("f", 2.0)
        plan.setPropertyInt("i", 2)
        plan.setPropertyUInt("u", 2)
        # Set arrays at once
        # RunPlan::setProperty(const std::string &name, const std::array<T, N> &value)
        plan.setPropertyArrayFloat("f_a", 3, (-2.0, 0.0, 2.0))
        plan.setPropertyArrayInt("i_a", 3, (-2, 0, 2))
        # Set individual elements at a time
        # RunPlan::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value)
        plan.setPropertyUInt("u_a", 0, 3)
        plan.setPropertyUInt("u_a", 1, 4)
        plan.setPropertyUInt("u_a", 2, 5)

        assert plan.getPropertyFloat("f") != environment.getPropertyFloat("f")
        assert plan.getPropertyInt("i") != environment.getPropertyInt("i")
        assert plan.getPropertyUInt("u") != environment.getPropertyUInt("u")
        assert plan.getPropertyArrayFloat("f_a") != environment.getPropertyArrayFloat("f_a")
        assert plan.getPropertyArrayInt("i_a") != environment.getPropertyArrayInt("i_a")
        assert plan.getPropertyArrayUInt("u_a") != environment.getPropertyArrayUInt("u_a")

        # Tests for exceptions
        # --------------------
        # Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
        # RunPlan::getProperty(const std::string &name) const
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyFloat("does_not_exist")
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyFloat("i")
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyUInt("u_a")
        assert e.value.type() == "InvalidEnvPropertyType"
        # std::array<T, N> RunPlan::getProperty(const std::string &name) const
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyArrayFloat("does_not_exist")
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyArrayFloat("u_a")
        assert e.value.type() == "InvalidEnvPropertyType"
        # Not a valid test when using getpropertyArray*
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        #     plan.getPropertyArrayInt("i_a")
        # assert e.value.type() == "InvalidEnvPropertyType"
        # with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
        #     plan.getPropertyArrayInt("i_a")
        # assert e.value.type() == "InvalidEnvPropertyType"
        # T RunPlan::getProperty(const std::string &name, const EnvironmentManager::size_type &index) const
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyFloat("does_not_exist", 0)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyFloat("u_a", 0)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            minus_one_uint32_t = -1 & 0xffffffff
            plan.getPropertyInt("i_a", minus_one_uint32_t)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.getPropertyInt("i_a", 4)
        assert e.value.type() == "OutOfBoundsException"

    def test_getRandomSimulationSeed(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Get the Simulation seed
        # As this is random, it could be any value. So get it twice, and make sure the same thing was returned?
        simulationSeed = plan.getRandomSimulationSeed()
        simulationSeedAgain = plan.getRandomSimulationSeed()
        assert simulationSeed == simulationSeedAgain
    def test_getSteps(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Get the default value
        steps = plan.getSteps()
        assert steps == 1

    def test_getOutputSubdirectory(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Get the default value
        subdir = plan.getOutputSubdirectory()
        # By default this is an empty string
        assert subdir == ""

    def test_getProperty(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Add some properties to the model, using a range of types.
        environment = model.Environment()
        environment.newPropertyFloat("f", 1.0)
        environment.newPropertyInt("i", -1)
        environment.newPropertyUInt("u", 1)
        environment.newPropertyArrayFloat("f_a", 3, (-1.0, 0.0, 1.0))
        environment.newPropertyArrayInt("i_a", 3, (-1, 0, 1))
        environment.newPropertyArrayUInt("u_a", 3, (0, 1, 2))
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Check that they match the original value when no overrides have been set.
        assert plan.getPropertyFloat("f") == environment.getPropertyFloat("f")
        assert plan.getPropertyInt("i") == environment.getPropertyInt("i")
        assert plan.getPropertyUInt("u") == environment.getPropertyUInt("u")
        assert plan.getPropertyArrayFloat("f_a") == environment.getPropertyArrayFloat("f_a")
        assert plan.getPropertyArrayInt("i_a") == environment.getPropertyArrayInt("i_a")
        assert plan.getPropertyArrayUInt("u_a") == environment.getPropertyArrayUInt("u_a")

        # Tests for exceptions
        # --------------------
        # Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
        # RunPlan::setProperty(const std::string &name, const T&value)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyFloat("does_not_exist", 1.)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyFloat("i", 1.)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyUInt("u_a", 1)
        assert e.value.type() == "InvalidEnvPropertyType"
        # RunPlan::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyArrayFloat("does_not_exist", 3, (2.0, 2.0, 2.0))
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyArrayFloat("u_a", 3, (2.0, 2.0, 2.0))
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyArrayInt("i_a", 2, (-2, 0))
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyArrayInt("i_a", 4, (-2, 0, 2, 2))
        assert e.value.type() == "InvalidEnvPropertyType"
        # RunPlan::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyFloat("does_not_exist", 0, 3.)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyFloat("u_a", 0, 3.)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            minus_one_uint32_t = -1 & 0xffffffff
            plan.setPropertyInt("i_a", minus_one_uint32_t, 3)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan.setPropertyInt("i_a", 4, 3)
        assert e.value.type() == "OutOfBoundsException"

    def test_operatorAssignment(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create two separate RunPlans with unique values
        plan1 = pyflamegpu.RunPlan(model)
        seed1 = 1
        plan1.setRandomSimulationSeed(seed1)
        plan2 = pyflamegpu.RunPlan(model)
        seedB = 2
        plan1.setRandomSimulationSeed(seedB)
        # Verify properties are unique
        assert plan1.getRandomSimulationSeed() != plan2.getRandomSimulationSeed()
        # use the assignment operator to set plan1=plan2, then check the unique value(s) are correct.
        plan1 = plan2
        assert plan1.getRandomSimulationSeed() == plan2.getRandomSimulationSeed()

    @pytest.mark.skip(reason="operator+ not currently wrapped")
    def test_operatorAddition(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create multiple run plans and set unique values on each
        plan1 = pyflamegpu.RunPlan(model)
        seed1 = 1
        plan1.setRandomSimulationSeed(seed1)
        plan2 = pyflamegpu.RunPlan(model)
        seed2 = 2
        plan2.setRandomSimulationSeed(seed2)
        plan3 = pyflamegpu.RunPlan(model)
        seed3 = 3
        plan3.setRandomSimulationSeed(seed3)

        # RunPlanVector = RunPlan + RunPlan
        vec12 = plan1 + plan2
        assert vec12.size(), 2
        assert vec12[0].getRandomSimulationSeed() == seed1
        assert vec12[1].getRandomSimulationSeed() == seed2

        # Try with operators in the other order.
        # As an append.
        vec123 = vec12 + plan3
        assert vec123.size(), 3
        assert vec123[0].getRandomSimulationSeed() == seed1
        assert vec123[1].getRandomSimulationSeed() == seed2
        assert vec123[2].getRandomSimulationSeed() == seed3

        # Expected exceptions
        # -------------------
        # Adding runplans together which are not for the same model (actually environment) should throw.
        otherModel = pyflamegpu.ModelDescription("other")
        otherModel.Environment().newPropertyFloat("f", 1.0)  # If both models have null environments they are compatible
        otherPlan = pyflamegpu.RunPlan(otherModel)
        otherPlanVector = pyflamegpu.RunPlanVector(otherModel, 1)

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            _ = plan1 + otherPlan
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            _ = plan1 + otherPlanVector
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            _ = otherPlan + plan1
        assert e.value.type() == "InvalidArgument"

    # RunPLanVector = RunPlan * uint32_t
    def test_operatorMultiplication(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an individual run plan.
        plan = pyflamegpu.RunPlan(model)
        # Set a value to a non default value to allow comparison
        newSimulationSeed = 12
        plan.setRandomSimulationSeed(newSimulationSeed)
        # Create a RunPlanVector of N elemets
        N = 4
        plans = plan * N
        assert plans.size(), N
        # Compare each element
        for p in plans:
            assert p.getRandomSimulationSeed() == newSimulationSeed
