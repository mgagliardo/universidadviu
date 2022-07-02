
NVCC        = nvcc -G -g
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = barnes1
OBJ	        = main.o

default: $(EXE)

main.o: main.cu kernel.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

backup:

	tar -cf `whoami`.tar kernel.cu main.cu 
