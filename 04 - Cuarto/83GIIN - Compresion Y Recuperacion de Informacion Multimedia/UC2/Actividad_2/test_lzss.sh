#!/bin/bash


for X in ./InputData/*
do
    if [ -f "$X" ]
    then
        echo checking $X
        filesize=$(stat -c '%s' $X)
        printf "uncompressed size:\t%d\n" $filesize
        ./lzss-master/lzss -c -i $X -o foo
        ./lzss-master/lzss -d -i foo -o bar
        diff $X bar
        filesize=$(stat -c '%s' foo)
        printf "compressed size:\t%d\n\n" $filesize
        rm foo
        rm bar
    fi
done

exit 0
