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
#include <thrust/inner_product.h>
#include <cmath>
#include <thrust/sort.h>
#include "GPUErrors.h"

#define TPB 256
#define K 500

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

void GenAdj(edge* edge_list, float* adj, unsigned int node_size, unsigned int edge_size);

void h_MatVecMult(float* h_A, float* h_x, float* h_y, unsigned int node_size);


//CUDA

__host__ void  RIM_rand_Ver1(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set);

__global__ void sparseCSRMat_Vec_Mult(unsigned int* csc, unsigned int* succ, float* values, float* vec, float* result, unsigned int node_size);

__global__ void Float_VectAdd(float* vec1, float* vec2, unsigned int size);

__global__ void Init_Random(float* vec, float* rand_init, unsigned int size, unsigned int k);

__host__ void Verify(float* gpu_vec, float* cpu_vec, unsigned int size);

__host__ void CheckSparseMatVec(unsigned int* csc, unsigned int* succ,edge* edge_list, unsigned int node_size, unsigned int edge_size);