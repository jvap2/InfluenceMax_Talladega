#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>   
#include<algorithm>
#include<cstdlib>
#include<ctime>
#include <cmath>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "GPUErrors.h"

#define TPB 256
#define K 10


#include "../include/GPUErrors.h"

typedef struct{
    unsigned int src;
    unsigned int dst;
    float weight;
}edge_t_IC;

typedef struct{
    unsigned int num;
    float prob;
}node_t_LT;

typedef struct{
    node_t_LT src;
    node_t_LT dst;
}edge_t_LT;

//CUDA Code 
__global__ void NodeSel_V1(unsigned int* hist_bin, unsigned int* RR_nodes, unsigned int size);
__host__ void IMM_Ver1();


//host code
__host__ float Calc_LogComb(unsigned int n, unsigned int k);

