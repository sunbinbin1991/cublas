#include <stdio.h>
//#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>
//#include "ctime"
//#include <thread>
//#include <vector>
//#include "cblas.h"
//#include <algorithm>
//#include <Windows.h>
using namespace std;
#define WIN32_LEAN_AND_MEAN
typedef struct
{
	unsigned int wa, ha, wb, hb, wc, hc;
} matrixSize;

void testdemo() {
	int const m = 5;//features_in_lib
	int const n = 3;//numtocom
	int const k = 2;//featurelen
	float *A;
	float *B;
	float *C;
	float *d_A;
	float *d_B;
	float *d_C;
	A = (float*)malloc(sizeof(float)*m*k);  //在内存中开辟空间
	B = (float*)malloc(sizeof(float)*n*k);  //在内存中开辟空间
	C = (float*)malloc(sizeof(float)*m*n);
	std::cout << "A=\n";
	for (int i = 0; i < m*k; i++) {
		A[i] = i;
		std::cout << A[i] << "\t";
		if (i != 0 && (i + 1) % k == 0) {
			std::cout << "\n";
		}
	}
	std::cout << "\n";
	std::cout << "B=\n";
	for (int i = 0; i < n*k; i++) {
		B[i] = i;
		std::cout << B[i] << "\t";
		if (i != 0 && (i + 1) % n == 0) {
			std::cout << "\n";
		}
	}
	std::cout << "\n";
	float alpha = 1.0;
	float beta = 0.0;
	cudaMalloc((void**)&d_A, sizeof(float)*m*k);
	cudaMalloc((void**)&d_B, sizeof(float)*n*k);
	cudaMalloc((void**)&d_C, sizeof(float)*m*n);
	//cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, k, alpha, B, k, A, k, beta, C, m);
	cudaMemcpy(d_A, A, sizeof(float)*m*k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(float)*n*k, cudaMemcpyHostToDevice);
	// cudaThreadSynchronize();
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaSetDevice(0);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
	//cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, k, &beta, d_C, m);
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, d_A, m, d_B, n, &beta, d_C, m);
	//cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m);
	cudaMemcpy(C, d_C, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	std::cout << "C=\n";
	for (int i = 0; i < m*n; i++) {
		std::cout << C[i] << "\t";
		if (i != 0 && (i + 1) % n == 0) {
			std::cout << "\n";
		}
	}
	free(A);
	free(B);
	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);
}

int main() {
	testdemo();
	return 0;
}
