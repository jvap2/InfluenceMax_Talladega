#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>   
#include<algorithm>
#include<cstdlib>
#include<ctime>
#include <cmath>
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

#define NUMSTRM 4

#define HOMO_PATH "../Graph_Data_Storage/homo.csv"
#define HOMO_DATA_PATH "../Graph_Data_Storage/homo_info.csv"

using namespace std;

struct edge{
    unsigned int src;
    unsigned int dst;
};

void readData(string filename, edge* edge_list);

void get_graph_info(string path, unsigned int* nodes, unsigned int* edges);

void genCSC(edge* edge_list, unsigned int* succ, unsigned int* csc, unsigned int node_size, unsigned int edge_size);

void genCSR(edge* edge_list, unsigned int* src, unsigned int* succ, unsigned int node_size, unsigned int edge_size);



//CUDA

__host__ void  RIM_rand_Ver1(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set);

__global__ void sparseCSRMat_Vec_Mult(unsigned int* csc, unsigned int* succ, float* vec, float* result, unsigned int node_size);

__global__ void Float_VectAdd(float* vec1, float* vec2, unsigned int size);

__global__ void Init_Random(float* vec, float* rand_init, unsigned int size, unsigned int k);