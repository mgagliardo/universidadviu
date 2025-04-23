#!/bin/bash

for X in ./InputData/*
do
    if [ -f "$X" ]
    then
        echo checking $X
        filesize=$(stat -c '%s' $X)
        printf "uncompressed size:\t\t%d\n" $filesize
        ./arcode-master/arcode -c -i $X -o foo
        filesize=$(stat -c '%s' foo)
        printf "static model compressed size:\t%d\n" $filesize
        ./arcode-master/arcode -d -i foo -o bar
        diff $X bar
        rm foo
        rm bar
        ./arcode-master/arcode -ca -i $X -o foo
        filesize=$(stat -c '%s' foo)
        printf "adaptive model compressed size:\t%d\n\n" $filesize
        ./arcode-master/arcode -da -i foo -o bar
        diff $X bar
        rm foo
        rm bar
    fi
done

exit 0
