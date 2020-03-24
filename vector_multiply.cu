#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>
#include <chrono>

const int a_row = 512;
const int a_col = 3136;
const int b_row = 3136;
const int b_col = 1;

//2维网格2维线程块  最常用
__global__ void VectorMultiply(float** a, float** b, float** c) {
    int thread_x_id = blockIdx.x * blockDim.x + threadIdx.x;   //x是线程块的x做行 线程的x做列
    int thread_y_id = blockIdx.y * blockDim.y + threadIdx.y;  //y是线程块的y做行 线程的y做列
    
    for (int k = 0; k < a_col; k++) {
        c[thread_y_id][thread_x_id] += (a[thread_y_id][k] + b[k][thread_x_id]); 
    }
}

int main() {
    std::chrono::system_clock::time_point begin = std::chrono::system_clock::now();
    //二维指针 指向一维指针（数据指针）
    float* a[a_row] = { NULL };
    float* b[b_row] = { NULL };
    float* c[a_row] = { NULL };
    float data_a[a_row * a_col] = { 0.0 };
    float data_b[b_row * b_col] = { 0.0 };
    float data_c[a_row * b_col] = { 0.0 };

    //二维指针 指向一维指针（数据指针）
    float** device_a = NULL;
    float** device_b = NULL;
    float** device_c = NULL;
    float* device_data_a = NULL;
    float* device_data_b = NULL;
    float* device_data_c = NULL;

    //分配显存
    cudaMalloc((void**)&device_a, sizeof(float*) * a_row);
    cudaMalloc((void**)&device_b, sizeof(float*) * b_row);
    cudaMalloc((void**)&device_c, sizeof(float*) * a_row);
    cudaMalloc((void**)&device_data_a, sizeof(float) * a_row * a_col);
    cudaMalloc((void**)&device_data_b, sizeof(float) * b_row * b_col);
    cudaMalloc((void**)&device_data_c, sizeof(float) * a_row * b_col);

    for (int i = 0; i < a_row * a_col; i++) {
        data_a[i] = 1;
    }
    for (int i = 0; i < b_row * b_col; i++) {
        data_b[i] = 1;
    }
    
    //主机二级指针存放着设备一级指针（数据指针）的地址
    //再通过cudacopy 传给设备二级指针 让设备二级指针和设备一级指针关联（数据指针）
    //而设备一级指针已经通过主机一级指针cudacopy了
    for (int i = 0; i < a_row; i++) {
        a[i] = device_data_a + i * a_col;
        c[i] = device_data_c + i * b_col;
    }
    for (int i = 0; i < b_row; i++) {
        b[i] = device_data_b + i * b_col;
    }

    //将内存中a和b数组的值复制到GPU中显存中
    cudaMemcpy(device_a, a, sizeof(float*) * a_row, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(float*) * b_row, cudaMemcpyHostToDevice);
    cudaMemcpy(device_c, c, sizeof(float*) * a_row, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data_a, data_a, sizeof(float) * a_row * a_col, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data_b, data_b, sizeof(float) * b_row * b_col, cudaMemcpyHostToDevice);

    //一个kernel函数由一个gpu的一个grid执行  
    //调用核函数 cpu调用 gpu运行 
    dim3 dim_block(32, 16);      //一个线程块block包含 512个线程threads(最多不超过512个)
    dim3 dim_grid((b_col + dim_block.x - 1) / dim_block.x, (a_row + dim_block.y - 1) / dim_block.y); //一个grid网格包含n / 512个线程块blocks(为了充分利用sm blocks尽可能多)
    VectorMultiply<<<dim_grid, dim_block>>>(device_a, device_b, device_c);

    //1级指针设备 给1级指针主机 赋值
    cudaMemcpy(data_c, device_data_c, sizeof(float) * a_row * b_col, cudaMemcpyDeviceToHost);
    
        for (int i = 0; i < a_row * a_col; i++) 
            printf("a[%d]=%.0f, ", i, data_a[i]);
        for (int i = 0; i < b_row * b_col; i++) 
            printf("b[%d]=%.0f, ", i, data_b[i]);
        for (int i = 0; i < a_row * b_col; i++) 
            printf("c[%d]=%.0f\n", i, data_c[i]);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //设置单位为秒
    std::chrono::duration<int, std::milli> milli = std::chrono::duration_cast<
                                                   std::chrono::milliseconds>(end - begin);
    printf("\n程序耗时：%dms\n", milli.count());
   

    //释放gpu显存
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFree(device_data_a);
    cudaFree(device_data_b);
    cudaFree(device_data_c);

    return 0;
}
