# GPU Teaching Kit: Accelerated Computing
# Probabilistic Bloom Filter Demo Project

Implementation of a bloom filter.

## Building bloom

There are five different executables that can be compiled. See the included
report for the differences between them.

    cd src

    make -f makefile_CPU
    make -f makefile_GPU
    make -f makefile_GPU2
    make -f makefile_OMP

    bin/pbfCPU
    bin/pbfGPU
    bin/pbfGPU2
    bin/pbfRNG
    bin/pbfOMP
