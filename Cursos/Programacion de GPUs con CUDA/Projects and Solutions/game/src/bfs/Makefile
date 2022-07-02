DEBUGDEFS = -DDEBUG

# ----- Make Macros -----

CXXFLAGS = -Wall -Wextra -Wshadow -pedantic -O3
CXX     = g++ $(DEBUGDEFS) -pg

NVCCFLAGS = -DCUDACC -DDEBUG -arch=sm_20 -O3 -Xcompiler -Wall -Xcompiler -Wextra
NVCC = nvcc

TARGETS = main

# All files for CPU version

CFILES = main.cpp bfs.cpp checkersboard.cpp
CUFILES = $(CFILES) bfskernels.cu

CPUOFILES = $(patsubst %.cpp, CPU/%.o, $(CFILES))
GPUOFILES = $(patsubst %.cpp, GPU/%.o, $(patsubst %.cu, GPU/%.o, $(CUFILES)))

# ----- Make Rules -----

all: CPU

checkersboard.cpp: getseq

CPU: cpudir $(CPUOFILES)
	$(CXX) $(CXXFLAGS) -o $@/$(TARGETS) $(CPUOFILES)

CPU/%.o: %.cpp 
	$(CXX) -o $@ -c $(CXXFLAGS) $<

cuda: GPU

GPU: gpudir $(GPUOFILES)
	$(NVCC) $(NVCCFLAGS) -o $@/$(TARGETS) $(GPUOFILES)

GPU/%.o: %.cpp 
	$(NVCC) -o $@ -c $(NVCCFLAGS) $<

GPU/%.o: %.cu
	$(NVCC) -o $@ -c $(NVCCFLAGS) $<

getseq:
	cp ../sequential/checkersboard.cpp . 

cpudir:
	mkdir -p CPU

gpudir:
	mkdir -p GPU

clean:
	rm CPU/* -f
	rm GPU/* -f
	rm checkersboard.cpp -f
	rm main.o* -f
	rm main.e* -f

