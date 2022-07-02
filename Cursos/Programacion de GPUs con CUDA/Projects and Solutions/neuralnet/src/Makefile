
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20 -Ifann/include -DFANN_NO_SEED
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE         = nn

BUILD_DIR   = build
FANN_PATH   = fann
FANN_SRC    = fann.c fann_cascade.c fann_error.c fann_io.c fann_train.c fann_train_data.c # fixedfann.c floatfann.c doublefann.c 
FANN_SRC_CU = kernel.cu main.cu support.cu
FANN_OBJ    = $(FANN_SRC:%.c=$(BUILD_DIR)/%.o) $(FANN_SRC_CU:%.cu=$(BUILD_DIR)/%.o)

default: $(EXE)

$(BUILD_DIR)/main.o: $(FANN_PATH)/main.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(BUILD_DIR)/support.o: $(FANN_PATH)/support.cu $(FANN_PATH)/support.h
	mkdir -p $(BUILD_DIR)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(BUILD_DIR)/kernel.o: $(FANN_PATH)/kernel.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(BUILD_DIR)/%.o: $(FANN_PATH)/%.c
	mkdir -p $(BUILD_DIR)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(EXE): $(FANN_OBJ)
	$(NVCC) $(FANN_OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) $(EXE).o* $(EXE).e* results*/ $(BUILD_DIR)


CONFIG1 = 
CONFIG2 = 10 400 500 500 500 500 500 500 500 500 400
CONFIG3 = 10 500 750 750 750 750 750 750 750 750 500
CONFIG4 = 10 500 1000 1000 1000 1000 1000 1000 1000 1000 500
CONFIG5 = 10 1000 2000 2000 2000 2000 2000 2000 2000 2000 1000
CONFIG6 = 10 1000 3000 3000 3000 3000 3000 3000 3000 3000 1000
CONFIG7 = 5 1000 4000 4000 4000 1000

CONFIG = ${CONFIG1}
WAIT = while true; do if [ -f nn.o* ]; then tail nn.*; break; fi; sleep 1; done

MODE = 1 # 1, 2, 3, 10, 11, 12, or 13

.PHONY:
runall: $(EXE)
	while true; do rm nn.* -f && submitjob ./nn $(MODE) $(CONFIG2) > /dev/null && $(WAIT); if [ `cat nn.o* | grep "Correct" | wc -l` -eq 1 ]; then break; fi; done
	while true; do rm nn.* -f && submitjob ./nn $(MODE) $(CONFIG3) > /dev/null && $(WAIT); if [ `cat nn.o* | grep "Correct" | wc -l` -eq 1 ]; then break; fi; done
	while true; do rm nn.* -f && submitjob ./nn $(MODE) $(CONFIG4) > /dev/null && $(WAIT); if [ `cat nn.o* | grep "Correct" | wc -l` -eq 1 ]; then break; fi; done
	while true; do rm nn.* -f && submitjob ./nn $(MODE) $(CONFIG5) > /dev/null && $(WAIT); if [ `cat nn.o* | grep "Correct" | wc -l` -eq 1 ]; then break; fi; done
	while true; do rm nn.* -f && submitjob ./nn $(MODE) $(CONFIG6) > /dev/null && $(WAIT); if [ `cat nn.o* | grep "Correct" | wc -l` -eq 1 ]; then break; fi; done
	while true; do rm nn.* -f && submitjob ./nn $(MODE) $(CONFIG7) > /dev/null && $(WAIT); if [ `cat nn.o* | grep "Correct" | wc -l` -eq 1 ]; then break; fi; done
