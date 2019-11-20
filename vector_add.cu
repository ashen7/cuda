#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>

const int N = 512 * 1024;

//线程组织模型  
//1. N个线程块    每个线程块1个线程  列向量 thread_id = blockIdx.x , blockDim.x是0 threadIdx.x也是0
//2. 1个线程块    每个线程块N个线程  行向量 thread_id = threadIdx.x  
//3. M个线程块    每个线程块N个线程  二维矩阵 M行N列 thread_id = blockIdx.x * blockDim.x + threadIdx.x
//4. M×N个线程块  每个线程块1个线程    thread_id = blockIdx.y * gridDim.x + blockIdx.x
//5. 1线程块      每个线程块M*N个线程  thread_id = threadIdx.y * blockDim.x + blockIdx.x 
//6. M×N个线程    每个线程块P×Q个线程 索引有两个维度(最常用)
//   thread_x_id = blockIdx.x * blockDim.x + threadIdx.x
//   thread_y_id = blockIdx.y * blockDim.y + threadIdx.y
__global__ void VectorAdd(float* a, float* b, float* c) {
    //一维网格和一维线程块 相当于二维矩阵 行是线程块数量  列是线程数量
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[thread_id] = a[thread_id] + b[thread_id];
}

int main() {
    int start = clock();
    float a[N] = { 0.0 };
    float b[N] = { 0.0 };
    float c[N] = { 0.0 };
    float* device_a = NULL;
    float* device_b = NULL;
    float* device_c = NULL;
    
    //分配显存
    cudaMalloc((void**)&device_a, sizeof(float) * N);
    cudaMalloc((void**)&device_b, sizeof(float) * N);
    cudaMalloc((void**)&device_c, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    
    //将内存中a和b数组的值复制到GPU中显存中
    cudaMemcpy(device_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, c, sizeof(float) * N, cudaMemcpyHostToDevice);

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 
    dim3 dim_grid(N / 512);   //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    dim3 dim_block(512);      //一个线程块block包含 512个线程threads(最多不超过512个)
    VectorAdd<<<dim_grid, dim_block>>>(device_a, device_b, device_c);

    //GPU计算任务完成后 将数据传输回CPU
    cudaMemcpy(c, device_c, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        printf("%.0f + %.0f = %.0f\t", a[i], b[i], c[i]);
        if ((i + 1) % 10 == 0)
            printf("\n");
    }
    int end = clock();
    printf("\n程序耗时：%ds\n", (end - start) / 1000);

    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}
