
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = nbody_block_cuda
OBJ	        = nbody_block_cuda.o

default: $(EXE)

nbody_block_cuda.o: nbody_block_cuda.cu 
	$(NVCC) -c -o $@ nbody_block_cuda.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

