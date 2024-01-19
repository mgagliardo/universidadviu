#! /bin/bash
#
gcc -c -Wall linpack_bench.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
gcc linpack_bench.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
#
rm linpack_bench.o
#
chmod ugo+x a.out
mv a.out ~/binc/linpack_bench
#
echo "Normal end of execution."
