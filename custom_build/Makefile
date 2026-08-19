


ARMCC = aarch64-none-linux-gnu-g++

CFLAGS = -Wall


OPTIM = -O2



SRC_DIRS = ./.. ./../transformer_layers ./../accelerator

INC_DIRS 	= 	./../ ./../transformer_layers ./../weights_data ./../accelerator/ 



SRC  = $(wildcard $(addsuffix /*.cc, $(SRC_DIRS)))
SRC += $(wildcard $(addsuffix /*.cpp, $(SRC_DIRS)))
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)


OUT_NAME ?= TicSat_3D


# CFLAGS += -DDEVELOP
CFLAGS += -DBWMA
CFLAGS += -DREARRANGE
CFLAGS += -DSA 
CFLAGS += -DSA_SIZE=8
CFLAGS += -DCORE_NUM=1

# CFLAGS += -DRELOAD_WEIGHT
# CFLAGS += -DNANO_3D
CFLAGS += -DN_3D_LAYERS=1
CFLAGS += -DSA_W=8
CFLAGS += -DSA_H=8



compile:
	$(ARMCC) $(CFLAGS) -march=armv8-a -lm $(OPTIM) $(SRC) $(INC) -o $(OUT_NAME) 



se_sim:
	cd launch_gem5/ && ./launch_gem5_se.sh ./out_dir stats config


launch-se:
	$(ARMCC) $(CFLAGS) -march=armv8-a -lm $(OPTIM) $(SRC) $(INC) -o $(OUT_NAME) -static
	make se_sim



rebuild-gem5:
	cd ./../gem5 && ./build_arm_gcc13.sh