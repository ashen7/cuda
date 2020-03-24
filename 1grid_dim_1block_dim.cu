#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>

const int M = 512;
const int N = 1024;

__global__ void VectorAdd(float* a, float* b, float* c) {
    //一维网格和一维线程块 相当于二维矩阵 行是线程块数量  列是线程数量
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("thread: %d\tblockIdx.x: %d\t, blockDim.x: %d\t, threadIdx.x: %d\n", thread_id, blockIdx.x, blockDim.x, threadIdx.x);
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
    dim3 dim_grid(M);       //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    dim3 dim_block(N);      //一个线程块block包含 512个线程threads(最多不超过512个)
    VectorAdd<<<dim_grid, dim_block>>>(device_a, device_b, device_c);

    //GPU计算任务完成后 将数据传输回CPU
    cudaMemcpy(c, device_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < M * N; i++) 
        printf("%.0f + %.0f = %.0f\t", a[i], b[i], c[i]);
    int end = clock();
    printf("\n程序耗时：%dms\n", (end - start) / 1000);

    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}
