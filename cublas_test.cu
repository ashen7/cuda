#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#define M 6 //6行
#define N 5 //5列
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/*一个把坐标（i, j）转化为cublas数据格式坐标的函数；
假设一个点位于第i行第j列，每一列的长度为ld，
则计算出来的列优先坐标就是 j*ld+i */

static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal(handle, n - q, &alpha, &m[IDX2C(p, q, ldm)], ldm);
    cublasSscal(handle, ldm - p, &beta, &m[IDX2C(p, q, ldm)], 1);
}

int main() {
    cudaError_t cudaStatu;    
    cublasStatus_t state;
    cublasHandle_t handle; //管理一个cublas上下文的句柄
    int i, j;
    float* devPtrA;
    float* a;
    a = (float*)malloc(M * N * sizeof(float));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            a[IDX2C(j, i, M)] = (float)(j * M + i + 1);
            printf ("%7.0f", a[IDX2C(j, i, M)]);
        }
        printf ("\n");
    }
    printf ("\n");

    cudaStatu = cudaMalloc((void**)&devPtrA, M*N*sizeof(float));
    if (cudaStatu != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    state = cublasCreate(&handle);
    if (state != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    state = cublasSetMatrix(M, N, sizeof(float), a, M, devPtrA, M);
    if (state != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    modify(handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    state = cublasGetMatrix(M, N, sizeof(float), devPtrA, M, a, M);
    if (state != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }

    free(a);
    cudaFree(devPtrA);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
}

