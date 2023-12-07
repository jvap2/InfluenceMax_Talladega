#pragma once
#include <iostream>
#include <iomanip>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

//GPU Error Functions
bool HandleCUDAError(cudaError_t t);//This handles the cuda error 
bool GetCUDARunTimeError();//This gets error from the GPU
bool HandleCUSparseError(cusparseStatus_t t);//This handles the cusparse error
bool HandleCUBLASError(cublasStatus_t t);//This handles the cublas error
void printCudaMemoryUsage();
//Any assignment, use the error handling everytime