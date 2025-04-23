#!/bin/bash

if [ -f foo ]
then
    echo this script requies use of a file foo
    exit 1
fi

if [ -f bar ]
then
    echo this script requies use of a file bar
    exit 1
fi

for X in *
do
    if [ -f "$X" ]
    then
        echo checking $X
        echo checking $X
        filesize=$(stat -c '%s' $X)
        printf "uncompressed size:\t%d\n" $filesize
        echo static model
        ../arcode -c -i $X -o foo
        filesize=$(stat -c '%s' foo)
        printf "static model compressed size:\t%d\n" $filesize
        ../arcode -d -i foo -o bar
        diff $X bar
        rm foo
        rm bar
        echo adaptive model
        ../arcode -ca -i $X -o foo
        filesize=$(stat -c '%s' foo)
        printf "adaptive model compressed size:\t%d\n\n" $filesize
        ../arcode -da -i foo -o bar
        diff $X bar
        rm foo
        rm bar
    fi
done

exit 0
