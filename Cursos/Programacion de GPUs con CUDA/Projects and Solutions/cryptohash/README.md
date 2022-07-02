# GPU Teaching Kit: Accelerated Computing
# Demo Project: GPU Cryptohash

## Build

### Sequential Version

    cd src
    make


### CUDA Version

    cd src
    make gpu

## Run

Where `'stringhash'` should be a 32-character hash preceded and followed by a single `'`.


### Sequential Version

    ./cpu_hash 'stringhash'

### CUDA Version

    ./cuda_mdh 'stringhash'
