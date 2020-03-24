#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

using std::cout;
using std::endl;

int main() {
    const int m = 3;
    const int n = 4;
    const int k = 2;
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[8] = {7, 8, 9, 10, 1, 2, 3, 4};
    float C[12] = {0.5};
    float* d_A;
    float* d_B;
    float* d_C;
    for (int i = 0; i < 6; i++) {
      cout << A[i] << "\t";      
    }
    cout << endl;
    for (int i = 0; i < 8; i++) {
      cout << B[i] << "\t";      
    }
    cout << endl;
    for (int i = 0; i < 12; i++) {
      cout << C[i] << "\t";      
    }
    cout << endl;

    float alpha = 1.0;
    float beta = 0.0;
    cudaMalloc((void**)&d_A, sizeof(float)*m*k);
    cudaMalloc((void**)&d_B, sizeof(float)*k*n);
    cudaMalloc((void**)&d_C, sizeof(float)*m*n);

    cudaMemcpy(d_A, A, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*k*n, cudaMemcpyHostToDevice);
    //blas 基本线性代数子程序 
    cublasHandle_t handle;
    cublasCreate(&handle);
    //gemm 通用矩阵相乘 handle A和B的行优先/列优先, 左行 左列/右行 右列 
    //c=alpha*op(A)*op(B)+beta*C 修正偏差 alpha=1 beta=0就行了
    
    //设备A的指针 a的lda 这是列优先 a的lda就是a的行 
    int lda = k;
    int ldb = n;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, ldb, d_A, lda, &beta, d_C, n);

    //设备A的指针 a的lda 这是行优先 a的lda就是a的列
    //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m);

    cudaMemcpy(C, d_C, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m * n; i++){
        std::cout <<C[i]<<"\t";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

