
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -I${CURDIR}/../sequential -arch=sm_20 --ptxas-options=-v
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = checkers
OBJ	        = main.o support.o bfs.o

default: $(EXE)

debug: NVCC_FLAGS += -DDEBUG
debug: default

main.o: main.cu kernel.cu support.h bfs.cpp
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

bfs.o: bfs.cpp bfs.hpp
	$(NVCC) -c -o $@ bfs.cpp $(NVCC_FLAGS)
	
$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

submit:
	tar -cf `whoami`.tar kernel.cu report.txt


