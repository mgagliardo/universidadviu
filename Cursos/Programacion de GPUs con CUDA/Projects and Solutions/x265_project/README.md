# GPU Teach Kit: Accelerated Computing
# x265 Motion Vector Estimation Acceleration

## Building

This project depends on a signficant amount of x265 encoder harness code, which can be acquired with the following steps

    hg clone https://bitbucket.org/mrj10/x265

Copy the contents of the `src` directory to `x265/source`

    cp src/* x265/source/.

Then build the code.

    cd x265/build/linux
    ./make-Makefiles.bash


## Running

Find the location of the test files `New_ProRes.yuv`.
Replace the `PATH` and the `LD_LIBRARY_PATH` with the corresponding paths on your system.
This may be done without installing.

    export PATH=/home/you/install/bin:$PATH
    export LD_LIBRARY_PATH=/home/you/install/lib:$LD_LIBRARY_PATH
    ./ece408_competition --input News_ProRes.yuv --input-res 3840x2160 --fps 23.98 /dev/null

