#!/bin/bash
make clean
make -j8
cd ..
rm out.txt
rm *.yaml
#mpirun -np 1 cuda-gdb ./bin/CoMD-mpi -e -t setfl -p Cu01.eam.alloy 
mpirun -np 1 ./bin/CoMD-mpi -e #-t setfl -p Cu01.eam.alloy #| tee out.txt -
#cd ./src-mpi
#cat ../out.txt | grep "blockid >" | cut -d' ' -f 3  | sort -g | uniq | wc -l
