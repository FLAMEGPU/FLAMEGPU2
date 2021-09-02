import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint

# Test the RunPlan interface
#     @pytest.mark.skip(reason="Not yet implemented")

    # Exit function condition which leads to 0 steps being allowed for plans within a vector.
class exitcond(pyflamegpu.HostFunctionConditionCallback):
    def test___init__(self):
        super().__init__()

    def test_run(self, FLAMEGPU):
        return pyflamegpu.EXIT

class TestRunPlanVector(TestCase):   
    def test_constructor(self): 
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Declare a pointer
        plans = None
        # Use New
        initialLength = 4
        plans = pyflamegpu.RunPlanVector(model, initialLength)
        assert plans != None
        assert plans.size() == initialLength
        # Run the destructor
        plans = None
    
    # Test setting the random property seed
    def test_setRandomPropertySeed(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Create a vector of plans
        totalPlans = 2
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Get the current random property seed. No sensible options to check this is an expected value.
        plans.getRandomPropertySeed()
        # Set the seed to a new value, and check that it was applied.
        newPropertySeed = 12
        plans.setRandomPropertySeed(newPropertySeed)
        # Check the new seed was applied correctly.
        assert plans.getRandomPropertySeed() == newPropertySeed
    
    def test_setSteps(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Get the original value of steps, storing them for later.
        originalValues = [1] * totalPlans
        for idx in range(totalPlans):
            plan = plans[idx]
            originalValues[idx] = plan.getSteps()
        
        # Set the number of steps
        newSteps = 12
        plans.setSteps(newSteps)
        # For Check each plan against the previous value(s)
        for idx in range(totalPlans):
            plan = plans[idx]
            assert plan.getSteps() == newSteps
            assert plan.getSteps() != originalValues[idx]
        
        # Expect an exception if setting the value to 0?
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            plans.setSteps(0)

        # If the model has an exit condition, then it will not throw.
        modelWithExit = pyflamegpu.ModelDescription("modelWithExit")
        f = exitcond()
        modelWithExit.addExitConditionCallback(f)
        plansWithExit = pyflamegpu.RunPlanVector(modelWithExit, 1)
        # Do not expect an exception iff allow_o_steps is set.
        plansWithExit.setSteps(0)
    
    def test_setOutputSubdirectory(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Set the new value
        newSubdir = "test"
        plans.setOutputSubdirectory(newSubdir)
        # For Check each plan against the previous value(s)
        for idx in range(totalPlans):
            plan = plans[idx]
            assert plan.getOutputSubdirectory() == newSubdir
            assert plan.getOutputSubdirectory() != ""
        

    def test_setProperty(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Add a few environment properties to the model.
        environment = model.Environment()
        fOriginal = 1.0
        iOriginal = 1
        u3Original = (0, 1, 2)
        d3Original = (0., 1., 2.)
        environment.newPropertyFloat("f", fOriginal)
        environment.newPropertyInt("i", iOriginal)
        environment.newPropertyArrayUInt("u3", 3, u3Original)
        environment.newPropertyArrayDouble("d3", 3, d3Original)
        # Create a vector of plans
        totalPlans = 2
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Uniformly set each property to a new value, then check it has been applied correctly.
        fNew = 2.0
        iNew = 2
        u3New = (3, 4, 5)
        d3New = (3., 4., 5.)
        # void RunPlanVector::setProperty(const std::string &name, const T &value) 
        plans.setPropertyFloat("f", fNew)
        plans.setPropertyInt("i", iNew)
        # Check setting full arrays
        # void RunPlanVector::setProperty(const std::string &name, const std::array<T, N> &value) 
        # Explicit type is required, to coerce the std::array. Might need partial template specialisation for  where the value is a stdarray of T?
        plans.setPropertyArrayUInt("u3", 3, u3New)
        # Check setting individual array elements
        # void RunPlanVector::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) 
        plans.setPropertyDouble("d3", 0, d3New[0])
        plans.setPropertyDouble("d3", 1, d3New[1])
        plans.setPropertyDouble("d3", 2, d3New[2])
        # Check values are as expected by accessing the properties from each plan
        for plan in plans:
            assert plan.getPropertyFloat("f") == fNew
            assert plan.getPropertyInt("i") == iNew
            assert plan.getPropertyArrayUInt("u3") == u3New
            assert plan.getPropertyArrayDouble("d3") == d3New
        

        # Tests for exceptions
        # --------------------
        # Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
        # void RunPlanVector::setProperty(const std::string &name, const T &value)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyFloat("does_not_exist", 1.)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyFloat("i", 1.)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyUInt("u3", 1)
        assert e.value.type() == "InvalidEnvPropertyType"
        # void RunPlanVector::setProperty(const std::string &name, const std::array<T, N> &value)
        # Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyArrayFloat("does_not_exist", 3, (2., 2., 2.))
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyArrayFloat("u3", 3, (2., 2., 2.))
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyArrayDouble("d3", 2, (-2, 0))
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyArrayDouble("d3", 4, (-2, 0, 2, 2))
        assert e.value.type() == "InvalidEnvPropertyType"
        # void RunPlanVector::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyFloat("does_not_exist", 0, 3)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyFloat("u3", 0, 3)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            minus_one_uint32_t = -1 & 0xffffffff
            plans.setPropertyDouble("d3", minus_one_uint32_t, 3)
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            plans.setPropertyDouble("d3", 4, 3)
    
    # Check that all values set lie within the min and max inclusive
    # @todo - should fp be [min, max) like when using RNG?
    def test_setPropertyUniformDistribution(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Add a few environment properties to the model.
        environment = model.Environment()
        fOriginal = 0.0
        iOriginal = 0
        u3Original = (0, 0, 0)
        environment.newPropertyFloat("f", fOriginal)
        environment.newPropertyInt("i", iOriginal)
        environment.newPropertyArrayUInt("u3", 3, u3Original)
        # Create a vector of plans
        totalPlans = 10
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # No need to seed the random, as this is a LERP rather than a random distribution.

        # Uniformly set each property to a new value, then check it has been applied correctly.
        fMin = 1.
        fMax = 100.
        iMin = 1
        iMax = 100
        u3Min = (1, 101, 201)
        u3Max = (100, 200, 300)
        # void setPropertyUniformDistribution(const std::string &name, const T &min, const T &max)
        plans.setPropertyUniformDistributionFloat("f", fMin, fMax)
        plans.setPropertyUniformDistributionInt("i", iMin, iMax)
        # Check setting individual array elements
        # void setPropertyUniformDistribution(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &max)
        plans.setPropertyUniformDistributionUInt("u3", 0, u3Min[0], u3Max[0])
        plans.setPropertyUniformDistributionUInt("u3", 1, u3Min[1], u3Max[1])
        plans.setPropertyUniformDistributionUInt("u3", 2, u3Min[2], u3Max[2])
        # Check values are as expected by accessing the properties from each plan
        for plan in plans:
            assert plan.getPropertyFloat("f") >= fMin
            assert plan.getPropertyFloat("f") <= fMax
            assert plan.getPropertyInt("i") >= iMin
            assert plan.getPropertyInt("i") <= iMax
            u3FromPlan = plan.getPropertyArrayUInt("u3")
            assert u3FromPlan[0] >= u3Min[0]
            assert u3FromPlan[0] <= u3Max[0]
            assert u3FromPlan[1] >= u3Min[1]
            assert u3FromPlan[1] <= u3Max[1]
            assert u3FromPlan[2] >= u3Min[2]
            assert u3FromPlan[2] <= u3Max[2]
        

        # Tests for exceptions
        # --------------------
        singlePlanVector = pyflamegpu.RunPlanVector(model, 1)
        # Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
        # void RunPlanVector::setPropertyUniformDistribution(const std::string &name, const T &min, const T &max)
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            singlePlanVector.setPropertyUniformDistributionFloat("f", 1., 100.)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyUniformDistributionFloat("does_not_exist", 1., 100.)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyUniformDistributionFloat("i", 1., 100.)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyUniformDistributionUInt("u3", 1, 100)
        assert e.value.type() == "InvalidEnvPropertyType"
        # void RunPlanVector::setPropertyUniformDistribution(const std::string &name, const EnvironmentManager::size_type
        # Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            singlePlanVector.setPropertyUniformDistributionUInt("u3", 0, 1, 100)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyUniformDistributionFloat("does_not_exist", 0, 1., 100.)
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plans.setPropertyUniformDistributionFloat("u3", 0, 1., 100.)
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            minus_one_uint32_t = -1 & 0xffffffff
            plans.setPropertyUniformDistributionUInt("u3", minus_one_uint32_t, 1, 100)
        with pytest.raises(RuntimeError) as e: # std::out_of_range
            plans.setPropertyUniformDistributionUInt("u3", 4, 1, 100)
    
    # Checking for uniformity of distribution would require a very large samples size.
    # As std:: is used, we trust the distribution is legit, and instead just check for min/max.
    def test_setPropertyUniformRandom(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Add a few environment properties to the model.
        environment = model.Environment()
        fOriginal = 1.0
        iOriginal = 1
        u3Original = (0, 1, 2)
        environment.newPropertyFloat("f", fOriginal)
        environment.newPropertyInt("i", iOriginal)
        environment.newPropertyArrayUInt("u3", 3, u3Original)
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Seed the RunPlanVector RNG for a deterministic test.
        plans.setRandomPropertySeed(1)

        # Uniformly set each property to a new value, then check it has been applied correctly.
        fMin = 1.
        fMax = 100.
        iMin = 1
        iMax = 100
        u3Min = (1, 101, 201)
        u3Max = (100, 200, 300)
        # void setPropertyUniformRandom(const std::string &name, const T &min, const T &Max)
        plans.setPropertyUniformRandomFloat("f", fMin, fMax)
        plans.setPropertyUniformRandomInt("i", iMin, iMax)
        # Check setting individual array elements
        # void setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &Max)
        plans.setPropertyUniformRandomUInt("u3", 0, u3Min[0], u3Max[0])
        plans.setPropertyUniformRandomUInt("u3", 1, u3Min[1], u3Max[1])
        plans.setPropertyUniformRandomUInt("u3", 2, u3Min[2], u3Max[2])
        # Check values are as expected by accessing the properties from each plan
        for plan in plans:
            # Floating point types are inclusive-exclusive [min, Max)
            assert plan.getPropertyFloat("f") >= fMin
            assert plan.getPropertyFloat("f") <  fMax
            # Integer types are mutually inclusive [min, Max]
            assert plan.getPropertyInt("i") >= iMin
            assert plan.getPropertyInt("i") <= iMax
            # Check array values are correct, Integers so mutually inclusive
            u3FromPlan = plan.getPropertyArrayUInt("u3")
            assert u3FromPlan[0] >= u3Min[0]
            assert u3FromPlan[0] <= u3Max[0]
            assert u3FromPlan[1] >= u3Min[1]
            assert u3FromPlan[1] <= u3Max[1]
            assert u3FromPlan[2] >= u3Min[2]
            assert u3FromPlan[2] <= u3Max[2]
        
    
    # It's non trivial to check for correct distirbutions, and we rely on std:: so we are going to trust it works as intended.
    # Instead, just check that the value is different than the original. As this is not guaranteed due to (seeded) RNG, just check that atleast one value is different.
    # Real property types only. Non-reals are a static assert.
    def test_setPropertyNormalRandom(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Add a few environment properties to the model.
        environment = model.Environment()
        fOriginal = 1.0
        d3Original = (0., 1., 2.)
        environment.newPropertyFloat("f", fOriginal)
        environment.newPropertyArrayDouble("d3", 3, d3Original)
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Seed the RunPlanVector RNG for a deterministic test.
        plans.setRandomPropertySeed(1)

        # Uniformly set each property to a new value, then check that atleast one of them is not the default.
        fMean = 1.
        fStddev = 100.
        d3Mean = (1., 101., 201.)
        d3Stddev = (100., 200., 300.)
        # void setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev)
        plans.setPropertyNormalRandomFloat("f", fMean, fStddev)
        # Check setting individual array elements
        # void setPropertyNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev)
        plans.setPropertyNormalRandomDouble("d3", 0, d3Mean[0], d3Stddev[0])
        plans.setPropertyNormalRandomDouble("d3", 1, d3Mean[1], d3Stddev[1])
        plans.setPropertyNormalRandomDouble("d3", 2, d3Mean[2], d3Stddev[2])
        fAtleastOneNonDefault = False
        d3AtleastOneNonDefault = [False, False, False]
        # Check values are as expected by accessing the properties from each plan
        for plan in plans:
            if (plan.getPropertyFloat("f") != fOriginal):
                fAtleastOneNonDefault = True
            d3FromPlan = plan.getPropertyArrayDouble("d3")
            if (d3FromPlan[0] != d3Original[0]):
                d3AtleastOneNonDefault[0] = True
            if (d3FromPlan[1] != d3Original[1]):
                d3AtleastOneNonDefault[1] = True
            if (d3FromPlan[2] != d3Original[2]):
                d3AtleastOneNonDefault[2] = True
        
        # assert that atleast one of each value is non-default.
        assert fAtleastOneNonDefault
        assert d3AtleastOneNonDefault[0]
        assert d3AtleastOneNonDefault[1]
        assert d3AtleastOneNonDefault[2]
    
    # It's non trivial to check for correct distirbutions, and we rely on std:: so we are going to trust it works as intended.
    # Instead, just check that the value is different than the original. As this is not guaranteed due to (seeded) RNG, just check that atleast one value is different.
    # Real property types only. Non-reals are a static assert.
    def test_setPropertyLogNormalRandom(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Add a few environment properties to the model.
        environment = model.Environment()
        fOriginal = 1.0
        d3Original = (0., 1., 2.)
        environment.newPropertyFloat("f", fOriginal)
        environment.newPropertyArrayDouble("d3", 3, d3Original)
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Seed the RunPlanVector RNG for a deterministic test.
        plans.setRandomPropertySeed(1)

        # Uniformly set each property to a new value, then check that atleast one of them is not the default.
        fMean = 1.
        fStddev = 100.
        d3Mean = (1., 101., 201.)
        d3Stddev = (100., 200., 300.)
        # void RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const T &mean, const T &stddev) 
        plans.setPropertyLogNormalRandomFloat("f", fMean, fStddev)
        # Check setting individual array elements
        # void RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev) 
        plans.setPropertyLogNormalRandomDouble("d3", 0, d3Mean[0], d3Stddev[0])
        plans.setPropertyLogNormalRandomDouble("d3", 1, d3Mean[1], d3Stddev[1])
        plans.setPropertyLogNormalRandomDouble("d3", 2, d3Mean[2], d3Stddev[2])
        fAtleastOneNonDefault = False
        d3AtleastOneNonDefault = [False, False, False]
        # Check values are as expected by accessing the properties from each plan
        for plan in plans:
            if (plan.getPropertyFloat("f") != fOriginal):
                fAtleastOneNonDefault = True
            d3FromPlan = plan.getPropertyArrayDouble("d3")
            if (d3FromPlan[0] != d3Original[0]):
                d3AtleastOneNonDefault[0] = True
            if (d3FromPlan[1] != d3Original[1]):
                d3AtleastOneNonDefault[1] = True
            if (d3FromPlan[2] != d3Original[2]):
                d3AtleastOneNonDefault[2] = True
        
        # assert that atleast one of each value is non-default.
        assert fAtleastOneNonDefault
        assert d3AtleastOneNonDefault[0]
        assert d3AtleastOneNonDefault[1]
        assert d3AtleastOneNonDefault[2]
    
    def test_setPropertyRandom(self): 
        # @todo - test setPropertyRandom
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Add a few environment properties to the model.
        environment = model.Environment()
        environment.newPropertyFloat("f", 1.0)
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
    
        # This method should be %ignored by swig - it should not exist. Expect an exception
        with pytest.raises(AttributeError):
            plans.setPropertyRandom
        
    
    # Test getting the random property seed
    def test_getRandomPropertySeed(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # repeatedly create run vectors, and get the property seed. Once we've found 2 that are  different, stop.
        # If a maximum number of tries is reached, then we error.
        maxGenerations = 8
        prevSeed = 0
        seed = 0
        for i in range(maxGenerations):
            # Create the vector
            plans = pyflamegpu.RunPlanVector(model, 1)
            seed = plans.getRandomPropertySeed()
            if i > 0:
                # the seed shouldn't be the same as the previous seed, but it might be so do not expect_.
                if prevSeed != seed:
                    # Break out the loop
                    break
            prevSeed = seed
        
        assert prevSeed != seed
    
    def test_size(self): 
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create run plan vectors of a number of sizes and check the value
        plans0 = pyflamegpu.RunPlanVector(model, 0)
        assert plans0.size() == 0
        plans1 = pyflamegpu.RunPlanVector(model, 1)
        assert plans1.size() == 1
        plans4 = pyflamegpu.RunPlanVector(model, 4)
        assert plans4.size() == 4
        plans64 = pyflamegpu.RunPlanVector(model, 64)
        assert plans64.size() == 64
    
    @pytest.mark.skip(reason="operator+ not currently wrapped")
    def test_operatorAddition(self): 
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create multiple unique plans which can be used to check order of plans.
        plan1 = pyflamegpu.RunPlan(model)
        seed1 = 1
        plan1.setRandomSimulationSeed(seed1)
        plan2 = pyflamegpu.RunPlan(model)
        seed2 = 2
        plan2.setRandomSimulationSeed(seed2)
        plan3 = pyflamegpu.RunPlan(model)
        seed3 = 3
        plan3.setRandomSimulationSeed(seed3)
        plan4 = pyflamegpu.RunPlan(model)
        seed4 = 4
        plan4.setRandomSimulationSeed(seed4)
        # RunPlanVector operator+(const RunPlan& rhs) const
        vec12 = plan1 + plan2
        vec123 = vec12 + plan3
        assert vec123.size() == 3
        assert vec123[0].getRandomSimulationSeed() == seed1
        assert vec123[1].getRandomSimulationSeed() == seed2
        assert vec123[2].getRandomSimulationSeed() == seed3
        # # Disabled, as operator+ is always push_back for performance reasons.
        # vec312 = plan3 + vec12
        # assert vec312.size() == 3
        # assert vec312[0].getRandomSimulationSeed() == seed3
        # assert vec312[1].getRandomSimulationSeed() == seed1
        # assert vec312[2].getRandomSimulationSeed() == seed2
        
        # RunPlanVector operator+(const RunPlanVector& rhs) const
        vec12 = plan1 + plan2
        vec34 = plan3 + plan4
        vec1234 = vec12 + vec34
        assert vec1234.size() == 4
        assert vec1234[0].getRandomSimulationSeed() == seed1
        assert vec1234[1].getRandomSimulationSeed() == seed2
        assert vec1234[2].getRandomSimulationSeed() == seed3
        assert vec1234[3].getRandomSimulationSeed() == seed4
        # # Disabled, as operator+ is always push_back for performance reasons.
        # pyflamegpu.RunPlanVector vec3412 = vec34 + vec12
        # assert vec3412.size() == 4
        # assert vec3412[0].getRandomSimulationSeed() == seed3
        # assert vec3412[1].getRandomSimulationSeed() == seed4
        # assert vec3412[2].getRandomSimulationSeed() == seed1
        # assert vec3412[3].getRandomSimulationSeed() == seed2
        
        # RunPlanVector& operator+=(const RunPlan& rhs)
        vec123 = plan1 + plan2
        vec123 += plan3
        assert vec123.size() == 3
        assert vec123[0].getRandomSimulationSeed() == seed1
        assert vec123[1].getRandomSimulationSeed() == seed2
        assert vec123[2].getRandomSimulationSeed() == seed3
        # += cannot have a plan on the lhs and a plan vector on the right.
        
        # RunPlanVector& operator+=(const RunPlanVector& rhs)
        vec1234 = plan1 + plan2
        vec1234 += (plan3 + plan4)
        assert vec1234.size() == 4
        assert vec1234[0].getRandomSimulationSeed() == seed1
        assert vec1234[1].getRandomSimulationSeed() == seed2
        assert vec1234[2].getRandomSimulationSeed() == seed3
        assert vec1234[3].getRandomSimulationSeed() == seed4
        

        # Expected exceptions
        # -------------------
        # Adding runplans together which are not for the same model (actually environment) should throw.
        planVector = pyflamegpu.RunPlanVector(model, 1)
        otherModel = pyflamegpu.ModelDescription("other")
        otherModel.Environment().newPropertyFloat("f", 1.0)  # If both models have null environments they are compatible
        otherPlan = pyflamegpu.RunPlan(otherModel)
        otherPlanVector = pyflamegpu.RunPlanVector(otherModel, 1)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan1 + otherPlan
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            otherPlan + plan1
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            plan1 + otherPlanVector
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            otherPlanVector + plan1
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            planVector + otherPlan
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            otherPlan + planVector
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            planVector += otherPlan
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            otherPlanVector += plan1
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            planVector += otherPlanVector
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            otherPlanVector += planVector
        assert e.value.type() == "InvalidArgument"
    
    # RunPlanVector operator*(const unsigned int& rhs) const
    def test_operatorMultiplication(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        assert plans.size() == totalPlans

        # Multiply the plan vector by a fixed size
        # RunPlanVector operator*(const unsigned int& rhs) const
        mult = 2
        morePlans = plans * mult
        expectedSize = mult * totalPlans
        assert morePlans.size() == expectedSize

        # multiply a plan in-place
        # RunPlanVector& operator*=(const unsigned int& rhs)
        plans *= mult
        assert plans.size() == expectedSize
    
    # operator[]
    def test_operatorSubscript(self): 
        # Define the simple model to use
        model = pyflamegpu.ModelDescription("test")
        # Create a vector of plans
        totalPlans = 4
        plans = pyflamegpu.RunPlanVector(model, totalPlans)
        # Check that each in-range element can be accessed
        prev = None
        for idx in range(totalPlans):
            plan = plans[idx]
            if idx > 0:
                assert plan != prev
            
            prev = plan
        
    