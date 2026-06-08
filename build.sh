#!/bin/bash
set -eux 

# to_do: use build.rs instead

nvcc -O3 -c src/kernel.cu -o kernel.o -Xcompiler -fPIC
ar rcs libcubuffer.a kernel.o
RUSTFLAGS="-L native=. -l static=cubuffer -l dylib=cudart -l dylib=stdc++" cargo run 