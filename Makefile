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
# Folders containing FLAMEGPU example files and templates
#BIN_DIR := ../../bin/x64/
BIN_DIR := bin/
#IDIR := ../../include/
# For now this will only work for x64 linux. 32x is uninportant as deprecated in CUDA 8.0 Other systems are currently not possible to test.
LDIR := ../../lib/x86_64-linux-gnu/
TEMPLATE := ../../FLAMEGPU/templates/
XSD_SCHEMA := ../../FLAMEGPU/schemas/

INPUT_DATA:=iterations/0.xml

IDIR := api/
SRC_GPU := gpu/
SRC_POP := pop/
SRC_SIM := sim/
SRC_MODEL:=model/

#OPENGL_FLAGS := -lglut -lGLEW -lGLU -lGL
#FLAMELIB := -I $(IDIR) -I $(SRC_) -I $(SRC_CUDA) -I $(SRC_VIZ) -I $(IDIR)GL/

################################################################################
#Generating Dynamic Code from FLAMEGPU Templates

XML_MODEL:=$(SRC_)XMLModelFile.xml

all: build

#XML_Validate:
#	xmllint --noout $(XML_MODEL) --schema $(XSD_SCHEMA)XMMLGPU.xsd 

#XSLTPREP: XML_Validate
#XSLTPREP:
#	xsltproc $(TEMPLATE)header.xslt  $(XML_MODEL)> $(SRC_CUDA)header.h 
#	xsltproc $(TEMPLATE)FLAMEGPU_kernals.xslt $(XML_MODEL) > $(SRC_CUDA)FLAMEGPU_kernals.cu
#	xsltproc $(TEMPLATE)io.xslt $(XML_MODEL) > $(SRC_CUDA)io.cu 
#	xsltproc $(TEMPLATE)simulation.xslt $(XML_MODEL) > $(SRC_CUDA)simulation.cu 
#	xsltproc $(TEMPLATE)main.xslt $(XML_MODEL) > $(SRC_CUDA)main.cu

################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= "/usr/local/cuda-7.5"

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-g++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     := -L$(LDIR)

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif


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
SMS ?= 20 30 35 37 50 52

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
	@echo "make build -> builds all executables in either release or debug        *"
	@echo "           ------------------------------------------------            *"
	@echo "make visualisation_mode -> builds executables in visualisation mode    *"
	@echo "                                                                       *"
	@echo "make console_mode -> builds executables in console mode                *"
	@echo "                                                                       *"
	@echo "make < .. > dbg='arg' -> builds in Release/Debug only                  *"
	@echo "                                'arg' -> 0 or 1 value                  *"
	@echo "                                                                       *"
	@echo "To run executables for console mode, run below command:                *"
	@echo "make run_console iter='arg'                                            *"
	@echo "           Note that without the 'arg', it only runs for 1 iteration   *"
	@echo "           ------------------------------------------------            *"   
	@echo "To run executables for visualisation/console mode, run below command:  *"
	@echo "make run_vis                                                           *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"                
	@echo "Alternatively, run the bash script stored in bin/x64. The iteration    *"
	@echo "default value in console mode is 10. You can simple change it by       *"
	@echo "entering a new value while running the ./*.sh file.                    *"
	@echo "                                                                       *"
	@echo "           ------------------------------------------------            *"
	@echo "Note: You can manualy change the location/name of the INPUT_DATA       *"
	@echo "                                                                       *" 
	@echo "************************************************************************"
 
build: Console_mode 

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif


main.o: main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

GPU_C_FILES := $(wildcard $(SRC_GPU)*.cpp)
GPU_CO_FILES := $(addprefix $(SRC_GPU),$(notdir $(GPU_C_FILES:.cpp=.o)))

SIM_C_FILES := $(wildcard $(SRC_SIM)*.cpp)
SIM_CO_FILES := $(addprefix $(SRC_SIM),$(notdir $(SIM_C_FILES:.cpp=.o)))

POP_C_FILES := $(wildcard $(SRC_POP)*.cpp)
POP_CO_FILES := $(addprefix $(SRC_POP),$(notdir $(POP_C_FILES:.cpp=.o)))

MODEL_C_FILES := $(wildcard $(SRC_MODEL)*.cpp)
MODEL_CO_FILES := $(addprefix $(SRC_MODEL),$(notdir $(MODEL_C_FILES:.cpp=.o)))

# MOZ: FIX THIS LATER, SO IT CAN COMPILES ALL TOGETHER
$(SRC_GPU)%.o: $(SRC_GPU)%.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

$(SRC_SIM)%.o: $(SRC_SIM)%.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

$(SRC_MODEL)%.o: $(SRC_MODEL)%.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<

$(SRC_POP)%.o: $(SRC_POP)%.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(DEBUG) $(GENCODE_FLAGS) $(FLAMELIB) -o $@ -c $<


Console_mode: BUILD_TYPE=$(Mode_TYPE)_Console
Console_mode: $(SIM_CO_FILES)  $(MODEL_CO_FILES)  $(POP_CO_FILES)  $(GPU_CO_FILES) main.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p $(BIN_DIR)$(BUILD_TYPE)
	$(EXEC) mv $@ $(BIN_DIR)$(BUILD_TYPE)
	#@echo ./$(BUILD_TYPE)/Console_mode ../../examples/Console_mode/$(INPUT_DATA) '$$'{1:-1}> $(BIN_DIR)Console_mode.sh
	#chmod +x $(BIN_DIR)Console_mode.sh


run: Console_mode
	#cd $(BIN_DIR) && ./Console_mode.sh $(iter)

clean:
	rm -f *.o
	find . -name '*.o' -delete

clobber: clean 

