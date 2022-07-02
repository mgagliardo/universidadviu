#!/bin/bash
make -j8
cd ..
rm out.txt
rm *.yaml
qsub src-mpi/comd.pbs
cd ./src-mpi
