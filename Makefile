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
LDIR := 

INPUT_DATA:=iterations/0.xml

IDIR := api/
SRC_GPU := gpu/
SRC_POP := pop/
SRC_SIM := sim/
SRC_MODEL:=model/

STD11     := -std=c++11
STD14     := -std=c++14
BOOST_LIB := -I/usr/include/boost
CUDA_LIB  := -I/usr/local/cuda/include

################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda-7.5"
# export CUDA_PATH=/usr/local/cuda-8.0

HOST_COMPILER := g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

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
	@echo "           ------------------------------------------------            *"
	@echo "make all -> builds all executables in either release or debug          *"
	@echo "           ------------------------------------------------            *"
	@echo "make console_mode -> builds executables in console mode                *"
	@echo "                                                                       *"
	@echo "make < .. > dbg='arg' -> builds in Release/Debug only                  *"
	@echo "                                'arg' -> 0 or 1 value                  *"
	@echo "                                                                       *"
	@echo "To run executables for console mode, run below command:                *"
	@echo "make run_console           
	@echo "************************************************************************"
 
all: Console_mode 

main.o: main.cpp
	$(EXEC) $(NVCC) $(DEBUG) $(STD11) $(BOOST_LIB)  $(CUDA_LIB)  $(GENCODE_FLAGS) -o $@ -c $<


GPU_C_FILES := $(wildcard $(SRC_GPU)*.cpp)
GPU_CO_FILES := $(addprefix $(SRC_GPU),$(notdir $(GPU_C_FILES:.cpp=.o)))

SIM_C_FILES := $(wildcard $(SRC_SIM)*.cpp)
SIM_CO_FILES := $(addprefix $(SRC_SIM),$(notdir $(SIM_C_FILES:.cpp=.o)))

POP_C_FILES := $(wildcard $(SRC_POP)*.cpp)
POP_CO_FILES := $(addprefix $(SRC_POP),$(notdir $(POP_C_FILES:.cpp=.o)))

MODEL_C_FILES := $(wildcard $(SRC_MODEL)*.cpp)
MODEL_CO_FILES := $(addprefix $(SRC_MODEL),$(notdir $(MODEL_C_FILES:.cpp=.o)))


$(SRC_GPU)%.o: $(SRC_GPU)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) $(CUDA_LIB)  -o $@ -c $<

$(SRC_SIM)%.o: $(SRC_SIM)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11)  -o $@ -c $<

$(SRC_MODEL)%.o: $(SRC_MODEL)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) -o $@ -c $<

$(SRC_POP)%.o: $(SRC_POP)%.cpp
	$(EXEC) $(HOST_COMPILER) $(DEBUG) $(STD11) $(BOOST_LIB) -o $@ -c $<


Console_mode: BUILD_TYPE=$(Mode_TYPE)_Console
Console_mode: $(MODEL_CO_FILES)  $(POP_CO_FILES)  $(SIM_CO_FILES)  $(GPU_CO_FILES) main.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(BUILD_TYPE)
	$(EXEC) mv $@ $(BIN_DIR)$(BUILD_TYPE)
	find . -name '*.gch' -delete
	@echo ./$(BUILD_TYPE)/Console_mode ../../$(INPUT_DATA) '$$'{1:-1}> $(BIN_DIR)Console_mode.sh
	chmod +x $(BIN_DIR)Console_mode.sh


run: Console_mode
	cd $(BIN_DIR) && ./Console_mode.sh $(iter)

clean:
	rm -f *.o
	find . -name '*.o' -delete
	find . -name '*.gch' -delete
	
clobber: clean 
	rm -r $(BIN_DIR)

