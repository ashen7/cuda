#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>

const int Row = 2 * 2;
const int Col = 2 * 2;

//2维网格2维线程块  最常用
__global__ void VectorAdd(float** a, float** b, float** c) {
    int thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;   //x是线程块的x做行 线程的x做列
    int thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;   //y是线程块的y做行 线程的y做列
    //printf("blockIdx.y/x: %d/%d\tblockDim.y/x: %d/%d\tthreadIdx.y/x: %d/%d\tthread_y/x: %d/%d\n", blockIdx.y, blockIdx.x, blockDim.y, blockDim.x, threadIdx.y, threadIdx.x, thread_y_id, thread_x_id);
    if (thread_x_id < Col && thread_y_id < Row) {
        c[thread_y_id][thread_x_id] = a[thread_y_id][thread_x_id] + b[thread_y_id][thread_x_id];
    }
}

int main() {
    int start = clock();
    //二维指针 指向一维指针（数据指针）
    float* a[Row] = { NULL };
    float* b[Row] = { NULL };
    float* c[Row] = { NULL };
    float data_a[Row * Col] = { 0.0 };
    float data_b[Row * Col] = { 0.0 };
    float data_c[Row * Col] = { 0.0 };

    //二维指针 指向一维指针（数据指针）
    float** device_a = NULL;
    float** device_b = NULL;
    float** device_c = NULL;
    float* device_data_a = NULL;
    float* device_data_b = NULL;
    float* device_data_c = NULL;

    //分配显存
    cudaMalloc((void**)&device_a, sizeof(float*) * Row);
    cudaMalloc((void**)&device_b, sizeof(float*) * Row);
    cudaMalloc((void**)&device_c, sizeof(float*) * Row);
    cudaMalloc((void**)&device_data_a, sizeof(float) * Row * Col);
    cudaMalloc((void**)&device_data_b, sizeof(float) * Row * Col);
    cudaMalloc((void**)&device_data_c, sizeof(float) * Row * Col);

    for (int i = 0; i < Row * Col; i++) {
        data_a[i] = i;
        data_b[i] = i;
    }
    
    //主机二级指针存放着设备一级指针（数据指针）的地址
    //再通过cudacopy 传给设备二级指针 让设备二级指针和设备一级指针关联（数据指针）
    //而设备一级指针已经通过主机一级指针cudacopy了
    for (int i = 0; i < Row; i++) {
        a[i] = device_data_a + i * Col;
        b[i] = device_data_b + i * Col;
        c[i] = device_data_c + i * Col;
    }

    //将内存中a和b数组的值复制到GPU中显存中
    cudaMemcpy(device_a, a, sizeof(float*) * Row, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(float*) * Row, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, c, sizeof(float*) * Row, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data_a, data_a, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data_b, data_b, sizeof(float) * Row * Col, cudaMemcpyHostToDevice);

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 
    dim3 dim_block(16, 32);      //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((Col + dim_block.x - 1) / dim_block.x, (Row + dim_block.y - 1) / dim_block.y); //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    VectorAdd<<<dim_grid, dim_block>>>(device_a, device_b, device_c);

    //1级指针设备 给1级指针主机 赋值
    cudaMemcpy(data_c, device_data_c, sizeof(float) * Row * Col, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < Row * Col; i++) 
        printf("%.0f + %.0f = %.0f\t", data_a[i], data_b[i], data_c[i]);
    int end = clock();
    printf("\n程序耗时：%ds\n", (end - start) / 1000);
   

    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(device_data_a);
    cudaFree(device_data_b);
    cudaFree(device_data_c);

    return 0;
}
