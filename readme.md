# CuBuffer

This repository provides a lightweight method to access custom CUDA kernels from Rust through a minimal, typesafe RAII wrapper (`CuBuffer`) for allocating and copying memory between the Host and the Device. A CUDA C shim layer wraps the underlying CUDA APIs and kernels, which is then statically linked directly into the Rust binary. 

Some example are included.
