#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>

const int M = 2 * 2;
const int N = 10 * 5;

//2维网格2维线程块
__global__ void VectorAdd(float* a, float* b, float* c) {
    int thread_id = blockIdx.y * N * gridDim.x + blockIdx.x * N + (threadIdx.y * blockDim.x + threadIdx.x);
    printf("thread: %d\tblockIdx.x: %d\tblockDim.x: %d\tthreadIdx.x: %d\tthreadIdx.y: %d\n", thread_id, blockIdx.x, blockDim.x, threadIdx.x, threadIdx.y);
    c[thread_id] = a[thread_id] + b[thread_id];
}

int main() {
    int start = clock();
    float a[M * N] = { 0.0 };
    float b[M * N] = { 0.0 };
    float c[M * N] = { 0.0 };
    float* device_a = NULL;
    float* device_b = NULL;
    float* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(float) * M * N);
    cudaMalloc((void**)&device_b, sizeof(float) * M * N);
    cudaMalloc((void**)&device_c, sizeof(float) * M * N);

    for (int i = 0; i < M * N; i++) {
        a[i] = i;
        b[i] = i;
    }
    
    //将内存中a和b数组的值复制到GPU中显存中
    cudaMemcpy(device_a, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 
    dim3 dim_grid(2, 2);       //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    dim3 dim_block(10, 5);  //一个线程块block包含 512个线程threads(最多不超过512个)
    VectorAdd<<<dim_grid, dim_block>>>(device_a, device_b, device_c);

    //GPU计算任务完成后 将数据传输回CPU
    cudaMemcpy(c, device_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < M * N; i++) 
        printf("%.0f + %.0f = %.0f\t", a[i], b[i], c[i]);
    int end = clock();
    printf("\n程序耗时：%ds\n", (end - start) / 1000);

    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}
