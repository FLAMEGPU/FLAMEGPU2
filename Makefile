################################################################################
#
# FLAME GPU Script for CUDA 7.5
#
# Copyright 2016 University of Sheffield.  All rights reserved.
#
# Authors : Dr Mozhgan Kabiri Chimeh, Dr Paul Richmond
# Contact : {m.kabiri-chimeh,p.richmond}@sheffield.ac.uk
#
# NOTICE TO USER:
#
# University of Sheffield retain all intellectual property and
# proprietary rights in and to this software and related documentation.
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation without an express license agreement from
# University of Sheffield is strictly prohibited.
#
# For terms of licence agreement please attached licence or view licence
# on www.flamegpu.com website.
#
################################################################################
# USAGE : make help
################################################################################
#
# Makefile project only supported on Linux Platforms
#
################################################################################

BIN_DIR := bin/
LDIR := ../../lib/x86_64-linux-gnu/

INPUT_DATA:=iterations/0.xml

IDIR := *.h
SRC_GPU := gpu/
SRC_POP := pop/
SRC_SIM := sim/
SRC_MODEL:=model/
SRC_RUNTIME := runtime/
SRC_CURVE := runtime/cuRVE/

STD11     := -std=c++11
STD14     := -std=c++14
BOOST_LIB := -I/usr/include/boost
CUDA_LIB  := -I/usr/local/cuda/include
PER_THREAD:= --default-stream per-thread 
LINE_INFO := -lineinfo

TEST_DIR := TEST

################################################################################

# Location of the CUDA Toolkit
# CUDA_PATH ?= "/usr/local/cuda-7.5"
# export CUDA_PATH=/usr/local/cuda-8.0

HOST_COMPILER := g++
NVCC          := nvcc -ccbin $(HOST_COMPILER) $(LINE_INFO) #$(PER_THREAD)
#NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m64
CCFLAGS     :=
LDFLAGS     := -L$(LDIR)

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      Mode_TYPE := Debug
else
      Mode_TYPE := Release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc

# Common includes for OPENGL
LIBRARIES := 

################################################################################

SAMPLE_ENABLED := 1

# Gencode arguments
SMS ?= 37 50 52

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
help:
	@echo "************************************************************************"
	@echo "*  Copyright 2016 University of Sheffield.  All rights reserved.       *"
	@echo "************************************************************************"
	@echo "make all -> builds all executables in either release or debug          *"
	@echo "           ------------------------------------------------            *"
	@echo "make FGPU -> builds executables for FGPU                               *"
	@echo "                                                                       *"
	@echo "make < .. > dbg='arg' -> builds in Release/Debug only                  *"
	@echo "                                'arg' -> 0 or 1 value                  *"
	@echo "           ------------------------------------------------            *"
	@echo "To run FGPU, run below command:                                        *"
	@echo "make run                                                               *"
	@echo "           ------------------------------------------------            *"
	@echo "To run Test units, run below command:                                  *"
	@echo "make run_BOOST_TEST TSuite='arg'                                       *"
	@echo "                          'arg' can be the name the specific unit test *"
	@echo "************************************************************************"
 
all: FGPU doxygen

doxygen:
	doxygen Doxyfile

main.o: main.cu
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<	
	
main_MAS.o: main_MAS.cu
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<	
	
device_functions.o: tests/device_functions.cu
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<	
test_all.o: tests/test_all.cpp
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<

GPU_C_FILES := $(wildcard $(SRC_GPU)*.cpp)
GPU_CO_FILES := $(addprefix $(SRC_GPU),$(notdir $(GPU_C_FILES:.cpp=.o)))

GPU_CU_FILES := $(wildcard $(SRC_GPU)*.cu)
GPU_CUO_FILES := $(addprefix $(SRC_GPU),$(notdir $(GPU_CU_FILES:.cu=.o)))

SIM_C_FILES := $(wildcard $(SRC_SIM)*.cpp)
SIM_CO_FILES := $(addprefix $(SRC_SIM),$(notdir $(SIM_C_FILES:.cpp=.o)))

POP_C_FILES := $(wildcard $(SRC_POP)*.cpp)
POP_CO_FILES := $(addprefix $(SRC_POP),$(notdir $(POP_C_FILES:.cpp=.o)))

MODEL_C_FILES := $(wildcard $(SRC_MODEL)*.cpp)
MODEL_CO_FILES := $(addprefix $(SRC_MODEL),$(notdir $(MODEL_C_FILES:.cpp=.o)))

RUNTIME_CU_FILES := $(wildcard $(SRC_RUNTIME)*.cu)
RUNTIME_CUO_FILES := $(addprefix $(SRC_RUNTIME),$(notdir $(RUNTIME_CU_FILES:.cu=.o)))

CURVE_CU_FILES := $(wildcard $(SRC_CURVE)*.cu)
CURVE_CUO_FILES := $(addprefix $(SRC_CURVE),$(notdir $(CURVE_CU_FILES:.cu=.o)))

$(SRC_GPU)%.o: $(SRC_GPU)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) $(CUDA_LIB)  -o $@ -c $<
	
$(SRC_GPU)%.o: $(SRC_GPU)%.cu
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<

$(SRC_SIM)%.o: $(SRC_SIM)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) $(CUDA_LIB) -o $@ -c $<

$(SRC_MODEL)%.o: $(SRC_MODEL)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) $(CUDA_LIB) -o $@ -c $<

$(SRC_POP)%.o: $(SRC_POP)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB) -o $@ -c $<
	
$(SRC_CURVE)%.o: $(SRC_CURVE)%.cu
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<

# excluded
$(SRC_RUNTIME)%.o: $(SRC_RUNTIME)%.cu
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -dc -o $@ -c $<
	
#$(EXEC) nvcc $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
#$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

# build
FGPU: BUILD_TYPE=$(Mode_TYPE)
FGPU: $(MODEL_CO_FILES)  $(POP_CO_FILES)  $(SIM_CO_FILES)  $(GPU_CO_FILES)  $(GPU_CUO_FILES) $(CURVE_CUO_FILES) $(RUNTIME_CUO_FILES) main.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -rdc=true -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(BUILD_TYPE)
	$(EXEC) mv $@ $(BIN_DIR)$(BUILD_TYPE)
	find . -name '*.gch' -delete
	@echo ./$(BUILD_TYPE)/FGPU ../../$(INPUT_DATA) '$$'{1:-1}> $(BIN_DIR)FGPU.sh
	chmod +x $(BIN_DIR)FGPU.sh

FGPU_MAS: BUILD_TYPE=$(Mode_TYPE)
FGPU_MAS: $(MODEL_CO_FILES)  $(POP_CO_FILES)  $(SIM_CO_FILES)  $(GPU_CO_FILES)  $(GPU_CUO_FILES) $(CURVE_CUO_FILES) $(RUNTIME_CUO_FILES) main_MAS.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -rdc=true -lineinfo -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(BUILD_TYPE)
	$(EXEC) mv $@ $(BIN_DIR)$(BUILD_TYPE)
	find . -name '*.gch' -delete
	@echo ./$(BUILD_TYPE)/FGPU_MAS ../../$(INPUT_DATA) '$$'{1:-1}> $(BIN_DIR)FGPU_MAS.sh
	chmod +x $(BIN_DIR)FGPU_MAS.sh

BOOST_TEST: $(MODEL_CO_FILES)  $(POP_CO_FILES)  $(SIM_CO_FILES)  $(GPU_CO_FILES) $(GPU_CUO_FILES) $(CURVE_CUO_FILES) $(RUNTIME_CUO_FILES) device_functions.o test_all.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -rdc=true -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(TEST_DIR)
	$(EXEC) mv $@ $(BIN_DIR)$(TEST_DIR)
	@echo ./$(TEST_DIR)/BOOST_TEST --log_level=message --run_test='$$'{1:-}> $(BIN_DIR)RUN_TEST.sh #--log_level=test_suite
	chmod +x $(BIN_DIR)RUN_TEST.sh
	find . -name '*.gch' -delete

# execute	
run_BOOST_TEST: BOOST_TEST 
	cd $(BIN_DIR) && ./RUN_TEST.sh $(TSuite)
		
run_MAS: FGPU_MAS
	cd $(BIN_DIR) && ./FGPU_MAS.sh $(iter)

run: FGPU
	cd $(BIN_DIR) && ./FGPU.sh $(iter)
	
# profiling
profile_timeline:
	cd $(BIN_DIR)Release && nvprof --unified-memory-profiling off -f -o timeline.prof ./FGPU_MAS 

profile_metric:
	cd $(BIN_DIR)Release && nvprof --unified-memory-profiling off --metrics achieved_occupancy,executed_ipc -f -o metrics.prof ./FGPU_MAS 

profile_analysis:	
	cd $(BIN_DIR)Release && nvprof --unified-memory-profiling off --analysis-metrics -f -o analysis.prof ./FGPU_MAS 

	
profile_full: profile_timeline profile_metric profile_analysis
	
# cleaning
clean:
	rm -f *.o
	find . -name '*.o' -delete
	find . -name '*.gch' -delete
	
clobber: clean 
	rm -r $(BIN_DIR)

#--events all --metrics all	

