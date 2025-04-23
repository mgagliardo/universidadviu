#!/bin/bash

for X in ./InputData/*
do
    if [ -f "$X" ]
    then
	file_name="${X##*/}"
        printf "Generando codigos Huffman:\t%s\n" $file_name 
        ./huffman-master/huffman -c -t -i $X -o ./OutputData/$file_name.csv 
    fi
done

exit 0
