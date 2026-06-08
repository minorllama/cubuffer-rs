#include <cuda_runtime.h>
#include <iostream>


// to_do: CUDA error checking

// some example kernels
__global__ void vadd_kernel(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void grab_thread_id(float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = idx;
    }
}

// basic api
extern "C" void cuda_free(void* ptr) {
    float* buffer;
    buffer = (float*)ptr;
    cudaFree(buffer);
}

extern "C" int cuda_copy_to_device(void* device_ptr, const void* host_ptr, size_t n) {
    cudaMemcpy(device_ptr, host_ptr, n, cudaMemcpyHostToDevice);
    return 0;
}

extern "C" int cuda_copy_to_host(void* host_ptr, const void* device_ptr, size_t n) {
    cudaMemcpy(host_ptr, device_ptr, n, cudaMemcpyDeviceToHost);
    return 0;
}

// some tests
extern "C" void cuda_test_alloc_f32(float* ptr, size_t N) {
    grab_thread_id <<<1, 256>>>(ptr, N);
    cudaDeviceSynchronize();
    //std::cout << std::flush;
}

extern "C" void cuda_test_alloc_i32(float* ptr, size_t N) {
    grab_thread_id <<<1, 256>>>(ptr, N);
    cudaDeviceSynchronize();

    /*int size = N * sizeof(int);
    int* host_data = (int*) malloc(size);
    cuda_copy_to_host(host_data, ptr, size);

    for(int i = 0; i < N; i++){
        std::cout << host_data[i] << " ";
    }
    std::cout << std::flush;*/ 
}

extern "C" void cuda_test_thread_id_f32(float* ptr, size_t N) {
    grab_thread_id <<<1, 256>>>(ptr, N);
    cudaDeviceSynchronize();
    //std::cout << std::flush;
}

extern "C" void* cuda_malloc(size_t N) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, N);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return ptr;
}


// wrapper function for rust
extern "C" void cuda_example_vadd(const float* a, const float* b, float* c, int N) {
    float *d_a, *d_b, *d_c;
    int size = N * sizeof(float);

    // CUDA runtime handles initialization on first call to a runtime function

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vadd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // memcp device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
