#!/bin/bash

for X in ./InputData/*
do
    if [ -f "$X" ]
    then
        printf "checking %s\n" $X
        filesize=$(stat -c '%s' $X)
        printf "uncompressed size:\t%d\n" $filesize
        ./huffman-master/huffman -c -i $X -o foo       # Codificación Huffman
        ./huffman-master/huffman -d -i foo -o bar      # Decodificación Huffman
        diff $X bar
        filesize=$(stat -c '%s' foo)
        printf "traditional size:\t%d\n" $filesize
        ./huffman-master/huffman -C -c -i $X -o foo    # Codificación Huffman Canonico
        ./huffman-master/huffman -C -d -i foo -o bar   # Decodificación Huffman Canonico
        diff $X bar
        filesize=$(stat -c '%s' foo)
        printf "canonical size:\t\t%d\n\n" $filesize

        rm foo
        rm bar
    fi
done

exit 0
