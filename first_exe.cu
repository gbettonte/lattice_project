#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda/atomic>
#include <chrono> // Per misurare il tempo

#define M 1024
#define N 1024
#define NUM_BLOCKS 1024
#define NUM_THREADS_PER_BLOCK 1024

//optimistic approach
__global__ void find_minimum_test(int *input, int *global_minimum);
__global__ void find_minimum_fix(int *input, int *global_minimum);
__global__ void find_minimum_opt(int *input, int *global_minimum);




int main() {
    
    // Get a different random number each time the program runs
    srand(time(0));
    cudaError_t err;

    
    // Declare and initialize global minimum on CPU
    int global_minimum_cpu[1]; global_minimum_cpu[0] = INT_MAX; std::cout << "Initial max number: " << global_minimum_cpu[0] << std::endl;
    
    
    // Allocate memory for global minimum on GPU; two variables: one for each approach
    int *global_minimum_gpu_opt; int *global_minimum_gpu_fix;
    cudaMalloc(&global_minimum_gpu_opt, sizeof(int)); cudaMalloc(&global_minimum_gpu_fix, sizeof(int));
    //The follwing lines can be useful to detect errors; If there is an error with the CUDA operation we get -1
    err = cudaGetLastError();
    if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        // Possibly: exit(-1) if program cannot continue....
    }

    //we assign the initial max value to the variables of gpu
    cudaMemcpy(global_minimum_gpu_opt, global_minimum_cpu, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(global_minimum_gpu_fix, global_minimum_cpu, sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
    
    // Declare and initialize input array on CPU; this is just an example; the input array can be whatever
    int input_array_cpu[M * N];
    for (int i = 0; i < M * N; i++) {
        input_array_cpu[i] = i + 2; 
    }

    // Allocate memory for input array on GPU
    int *input_array_gpu;
    cudaMalloc(&input_array_gpu, M * N * sizeof(int));
        err = cudaGetLastError();
    if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(input_array_gpu, input_array_cpu, M * N * sizeof(int), cudaMemcpyHostToDevice); 
    err = cudaGetLastError();
    if ( err != cudaSuccess )
     {
        printf("here CUDA Error: %s\n", cudaGetErrorString(err));       
    }
    cudaDeviceSynchronize();

    //We loop and we a lot of time and aunch kernel; at the end we converge and we get the minimum
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++) {

        find_minimum_opt<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(input_array_gpu, global_minimum_gpu_opt);
        err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }
        cudaDeviceSynchronize();

    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // Copy result back to CPU
    cudaMemcpy(global_minimum_cpu, global_minimum_gpu_opt, sizeof(int), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    // Output the result
    std::cout << "Computed minimum value by the optimistic approach: " << global_minimum_cpu[0] << "  , in "<< duration.count() << "  seconds" <<std::endl;
    
    
    start = std::chrono::high_resolution_clock::now();
    find_minimum_fix<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(input_array_gpu, global_minimum_gpu_fix);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cudaMemcpy(global_minimum_cpu, global_minimum_gpu_opt, sizeof(int), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    std::cout << "Computed minimum value by fix point approach: " << global_minimum_cpu[0] << "  , in "<< duration.count() << "  seconds" <<std::endl;

    //The following is just a sligtly different implementation which exploit the synchronization between threads to find the minimum
    //I left it on purpose because I think it can be interesting
    start = std::chrono::high_resolution_clock::now();
    find_minimum_test<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(input_array_gpu, global_minimum_gpu_fix);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    cudaMemcpy(global_minimum_cpu, global_minimum_gpu_opt, sizeof(int), cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
    std::cout << "Computed minimum value by test: " << global_minimum_cpu[0] << "  , in "<< duration.count() << "  seconds" <<std::endl;

    // Free GPU memory
    cudaFree(global_minimum_gpu_opt);cudaFree(global_minimum_gpu_fix);
    cudaFree(input_array_gpu);
}

//The following is just a sligtly different implementation which exploit the synchronization between threads to find the minimum
//I left it on purpose because I think it can be interesting
__global__ void find_minimum_test(int *input, int *global_minimum) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    for(int i = 0; i < 100000; i++){
        if (input[index] < global_minimum[0])
            global_minimum[0] = input[index];
        __syncthreads();
    }
    //atomicExch(global_minimum, input[index]);
}
//fix approach
__global__ void find_minimum_fix(int *input, int *global_minimum) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (input[index] < global_minimum[0])
        atomicExch(global_minimum, input[index]);
}


//optimistic approach
__global__ void find_minimum_opt(int *input, int *global_minimum) {
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (input[index] < global_minimum[0]) 
        global_minimum[0] = input[index]; 
}
