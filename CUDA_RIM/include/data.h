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
#include <thrust/transform_reduce.h>
#include <thrust/set_operations.h>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "GPUErrors.h"

#define TPB 256
#define K 100

#define NUMSTRM 12

#define HOMO_PATH "../Graph_Data_Storage/homo.csv"
#define HOMO_DATA_PATH "../Graph_Data_Storage/homo_info.csv"
#define SEED_PATH "../RIM_res/res_4000.csv"
#define ARVIX_PATH "../Graph_Data_Storage/ca-GrQc.csv"
#define ARVIX_DATA_PATH "../Graph_Data_Storage/ca-GrQc-data.csv"
#define ARVIX_SEED_PATH "../RIM_res/res_arvix.csv"
#define WIKI_VOTE_PATH "../Graph_Data_Storage/wikivote.csv"
#define WIKI_VOTE_DATA_PATH "../Graph_Data_Storage/wikivote_data.csv"
#define WIKI_VOTE_SEED_PATH "../RIM_res/res_wiki.csv"
#define HEPTH_PATH "../Graph_Data_Storage/ca-HepTh.csv"
#define HEPTH_DATA_PATH "../Graph_Data_Storage/ca-HepTh-data.csv"
#define HEPTH_SEED_PATH "../RIM_res/res_HepTh.csv"
#define HEPTH_DATA_MEASURE "../RIM_data/HepTh/meas.csv"
#define WIKI_VOTE_DATA_MEASURE "../RIM_data/wiki-vote/meas.csv"
#define ARVIX_DATA_MEASURE "../RIM_data/arvix/meas.csv"
#define HOMO_DATA_MEASURE "../RIM_data/syn/meas.csv"
#define HEPTH_DATA_MEASURE_2 "../RIM_data/HepTh/meas_2.csv"
#define WIKI_VOTE_DATA_MEASURE_2 "../RIM_data/wiki-vote/meas_2.csv"
#define ARVIX_DATA_MEASURE_2 "../RIM_data/arvix/meas_2.csv"
#define HOMO_DATA_MEASURE_2 "../RIM_data/syn/meas_2.csv"

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

void Normalize_L2(float* h_x, unsigned int node_size);

void Export_Seed_Set_to_CSV(unsigned int* seed_set, unsigned int seed_size, string path);


__host__ void Save_Data(string file, float time, float damping_factor, float threshold,unsigned int epoch);


//CUDA

__host__ void  RIM_rand_Ver1(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file);

__host__ void  RIM_rand_Ver2(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file);

__host__ void  RIM_rand_Ver3_PR(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file);

__global__ void sparseCSRMat_Vec_Mult(unsigned int* csc, unsigned int* succ, float* values, float* vec, float* result, unsigned int node_size);

__global__ void Float_VectAdd(float* vec1, float* vec2, unsigned int size);

__global__ void Init_Random(float* vec, float* rand_init, unsigned int size, unsigned int k);

__host__ void Verify(float* gpu_vec, float* cpu_vec, unsigned int size);

__host__ void CheckSparseMatVec(unsigned int* csc, unsigned int* succ,edge* edge_list, unsigned int node_size, unsigned int edge_size);

__host__ void PageRank(float* pr_vector, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time);

__global__ void Init_Pr(float* pr_vector, unsigned int node_size);

__global__ void Gen_P_Mem_eff(float* weight_P, unsigned int* src, unsigned int* succ, unsigned int node_size, float* damp);

__global__ void Init_P(float* P, unsigned int node_size, float* damp);