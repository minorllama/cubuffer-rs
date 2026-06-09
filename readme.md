# CuBuffer

This repository provides a lightweight method to access custom CUDA kernels from Rust through a minimal, typesafe RAII wrapper (`CuBuffer`) for allocating and copying memory between the Host and the Device. A CUDA C shim layer wraps the underlying CUDA APIs and kernels, which is then statically linked directly into the Rust binary. 

## Basic example

Declare the unsafe `CUDA C` kernel. `unsafe { fn cuda_test_thread_id_f32(ptr: *mut f32, n:usize); }` with definition in `CUDA C` source,
```C++
    ...
    __global__ void grab_thread_id(float* c, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            c[idx] = idx;
        }
    } 
    ...
    extern "C" void cuda_test_thread_id_f32(float* ptr, size_t N) {
        grab_thread_id <<<1, 256>>>(ptr, N);
        cudaDeviceSynchronize();
    }
    ...
```
Warp that in a safe function: `pub fn cubuffer_test(buffer: &mut CuBuffer<f32>, size:usize){ unsafe { cuda_test_thread_id_f32(buffer.as_mut_ptr(), size); };}`. Build a static library. 
```bash
    nvcc -O3 -c src/kernel.cu -o kernel.o -Xcompiler -fPIC
    ar rcs libcubuffer.a kernel.o
```

Link into the rust binary built with with `cargo`, passing flags for linking against the built static library, and also agianst `stdc++`, `cudart`, with `L native=. -l static=cubuffer -l dylib=cudart -l dylib=stdc++`, 

```bash
    RUSTFLAGS="-L native=. -l static=cubuffer -l dylib=cudart -l dylib=stdc++" cargo run 
```

To use from rust source,
```rust
    use cubuffer::cubufferapi::*;
    
    ...
    
    // declare a buffer on cpu
    let size:usize = 8;
    let host_buffer_a:Vec<f32> = vec![1 as f32; size]; 
    println!("{:?}", host_buffer_a);
    
    // declare a buffer on device
    let mut buffer = CuBuffer::<f32>::new(size);
    // copy from cpu
    buffer.from_host(&host_buffer_a);
    
    // call the kernel
    cubuffer_test(&mut buffer, size);

    // copy back to a host buffer
    let mut host_buffer_b:Vec<f32> = vec![0 as f32; size];
    buffer.to_host(&mut host_buffer_b);
    eprintln!("{:?}", host_buffer_b); 
```

Also see [kernel.cu](src/kernel.cu), [cubufferapi.rs](src/cubufferapi.rs) and the cargo build script [build.rs](build.rs). 

## Non-trivial examples

Upcoming 




    



