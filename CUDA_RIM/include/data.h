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
#include <thrust/extrema.h>
#include "GPUErrors.h"

#define TPB 256
#define K 150

#define NUMSTRM 10

#define HOMO_PATH "../Graph_Data_Storage/homo.csv"
#define HOMO_DATA_PATH "../Graph_Data_Storage/homo_info.csv"
#define SEED_PATH "../RIM_res/res_4000.csv"
#define ARVIX_PATH "../Graph_Data_Storage/ca-GrQc.csv"
#define ARVIX_DATA_PATH "../Graph_Data_Storage/ca-GrQc-data.csv"
#define ARVIX_SEED_PATH "../RIM_res/res_arvix.csv"
#define WIKI_VOTE_PATH "../Graph_Data_Storage/wikivote.csv"
#define WIKI_VOTE_DATA_PATH "../Graph_Data_Storage/wikivote_data.csv"
#define WIKI_VOTE_SEED_PATH "../RIM_res/res_wiki.csv"
#define EP_PATH "../Graph_Data_Storage/epinions.csv"
#define EP_DATA_PATH "../Graph_Data_Storage/epinions_data.csv"
#define EP_SEED_PATH "../RIM_res/res_ep.csv"
#define HEPTH_PATH "../Graph_Data_Storage/ca-HepTh.csv"
#define HEPTH_DATA_PATH "../Graph_Data_Storage/ca-HepTh-data.csv"
#define HEPTH_SEED_PATH "../RIM_res/res_HepTh.csv"
#define HEPTH_DATA_MEASURE "../RIM_data/HepTh/meas.csv"
#define WIKI_VOTE_DATA_MEASURE "../RIM_data/wiki-vote/meas.csv"
#define ARVIX_DATA_MEASURE "../RIM_data/arvix/meas.csv"
#define HOMO_DATA_MEASURE "../RIM_data/syn/meas.csv"
#define EP_DATA_MEASURE "../RIM_data/epinions/meas.csv"
#define HEPTH_DATA_MEASURE_2 "../RIM_data/HepTh/meas_2.csv"
#define WIKI_VOTE_DATA_MEASURE_2 "../RIM_data/wiki-vote/meas_2.csv"
#define ARVIX_DATA_MEASURE_2 "../RIM_data/arvix/meas_2.csv"
#define HOMO_DATA_MEASURE_2 "../RIM_data/syn/meas_2.csv"
#define EP_DATA_MEASURE_2 "../RIM_data/epinions/meas_2.csv"
#define HEPTH_DATA_MEASURE_PR "../RIM_data/HepTh/meas_3.csv"
#define WIKI_VOTE_DATA_MEASURE_PR "../RIM_data/wiki-vote/meas_3.csv"
#define ARVIX_DATA_MEASURE_PR "../RIM_data/arvix/meas_3.csv"
#define HOMO_DATA_MEASURE_PR "../RIM_data/syn/meas_3.csv"
#define EP_DATA_MEASURE_PR "../RIM_data/epinions/meas_3.csv"
#define HEPTH_DATA_MEASURE_GREEDY "../RIM_data/HepTh/meas_4.csv"
#define WIKI_VOTE_DATA_MEASURE_GREEDY "../RIM_data/wiki-vote/meas_4.csv"
#define ARVIX_DATA_MEASURE_GREEDY "../RIM_data/arvix/meas_4.csv"
#define HOMO_DATA_MEASURE_GREEDY "../RIM_data/syn/meas_4.csv"
#define EP_DATA_MEASURE_GREEDY "../RIM_data/epinions/meas_4.csv"
#define HEPTH_DATA_MEASURE_SIG "../RIM_data/HepTh/meas_5.csv"
#define WIKI_VOTE_DATA_MEASURE_SIG "../RIM_data/wiki-vote/meas_5.csv"
#define ARVIX_DATA_MEASURE_SIG "../RIM_data/arvix/meas_5.csv"
#define HOMO_DATA_MEASURE_SIG "../RIM_data/syn/meas_5.csv"
#define EP_DATA_MEASURE_SIG "../RIM_data/epinions/meas_5.csv"
#define HEPTH_DATA_MEASURE_TANH "../RIM_data/HepTh/meas_6.csv"
#define WIKI_VOTE_DATA_MEASURE_TANH "../RIM_data/wiki-vote/meas_6.csv"
#define ARVIX_DATA_MEASURE_TANH "../RIM_data/arvix/meas_6.csv"
#define HOMO_DATA_MEASURE_TANH "../RIM_data/syn/meas_6.csv"
#define EP_DATA_MEASURE_TANH "../RIM_data/epinions/meas_6.csv"
#define HEPTH_PR "../Graph_Data_Storage/hepth_pr.csv"
#define WIKI_VOTE_PR "../Graph_Data_Storage/wikivote_pr.csv"
#define ARVIX_PR "../Graph_Data_Storage/arxiv_pr.csv"
#define HOMO_PR "../Graph_Data_Storage/homo_pr.csv"
#define EP_PR "../Graph_Data_Storage/epinions_pr.csv"
#define AM_PR "../Graph_Data_Storage/amazon_pr.csv"
#define AM_DATA_MEASURE_GREEDY "../RIM_data/amazon/meas_4.csv"
#define AM_DATA_MEASURE_SIG "../RIM_data/amazon/meas_5.csv"
#define AM_DATA_MEASURE_TANH "../RIM_data/amazon/meas_6.csv"
#define AM_DATA_MEASURE_PR "../RIM_data/amazon/meas_3.csv"
#define AM_DATA_MEASURE_2 "../RIM_data/amazon/meas_2.csv"
#define AM_DATA_MEASURE "../RIM_data/amazon/meas.csv"
#define AM_PATH "../Graph_Data_Storage/amazon.csv"
#define AM_DATA_PATH "../Graph_Data_Storage/amazon-data.csv"
#define AM_SEED_PATH "../RIM_res/res_amazon.csv"
#define ND_PR "../Graph_Data_Storage/nd_pr.csv"
#define ND_DATA_MEASURE_GREEDY "../RIM_data/nd/meas_4.csv"
#define ND_DATA_MEASURE_SIG "../RIM_data/nd/meas_5.csv"
#define ND_DATA_MEASURE_TANH "../RIM_data/nd/meas_6.csv"
#define ND_DATA_MEASURE_PR "../RIM_data/nd/meas_3.csv"
#define ND_DATA_MEASURE_2 "../RIM_data/nd/meas_2.csv"
#define ND_DATA_MEASURE "../RIM_data/nd/meas.csv"
#define ND_PATH "../Graph_Data_Storage/nd.csv"
#define ND_DATA_PATH "../Graph_Data_Storage/nd-data.csv"
#define ND_SEED_PATH "../RIM_res/res_nd.csv"
#define BRK_PR "../Graph_Data_Storage/berk_pr.csv"
#define BRK_DATA_MEASURE_GREEDY "../RIM_data/berk/meas_4.csv"
#define BRK_DATA_MEASURE_SIG "../RIM_data/berk/meas_5.csv"
#define BRK_DATA_MEASURE_TANH "../RIM_data/berk/meas_6.csv"
#define BRK_DATA_MEASURE_PR "../RIM_data/berk/meas_3.csv"
#define BRK_DATA_MEASURE_2 "../RIM_data/berk/meas_2.csv"
#define BRK_DATA_MEASURE "../RIM_data/berk/meas.csv"
#define BRK_PATH "../Graph_Data_Storage/berk.csv"
#define BRK_DATA_PATH "../Graph_Data_Storage/berk-data.csv"
#define BRK_SEED_PATH "../RIM_res/res_berk.csv"
#define GGL_PR "../Graph_Data_Storage/google_pr.csv"
#define GGL_DATA_MEASURE_GREEDY "../RIM_data/google/meas_4.csv"
#define GGL_DATA_MEASURE_SIG "../RIM_data/google/meas_5.csv"
#define GGL_DATA_MEASURE_TANH "../RIM_data/google/meas_6.csv"
#define GGL_DATA_MEASURE_PR "../RIM_data/google/meas_3.csv"
#define GGL_DATA_MEASURE_2 "../RIM_data/google/meas_2.csv"
#define GGL_DATA_MEASURE "../RIM_data/google/meas.csv"
#define GGL_PATH "../Graph_Data_Storage/google.csv"
#define GGL_DATA_PATH "../Graph_Data_Storage/google-data.csv"
#define GGL_SEED_PATH "../RIM_res/res_google.csv"
#define WKT_PR "../Graph_Data_Storage/wiki_talk_pr.csv"
#define WKT_DATA_MEASURE_GREEDY "../RIM_data/wiki_talk/meas_4.csv"
#define WKT_DATA_MEASURE_SIG "../RIM_data/wiki_talk/meas_5.csv"
#define WKT_DATA_MEASURE_TANH "../RIM_data/wiki_talk/meas_6.csv"
#define WKT_DATA_MEASURE_PR "../RIM_data/wiki_talk/meas_3.csv"
#define WKT_DATA_MEASURE_2 "../RIM_data/wiki_talk/meas_2.csv"
#define WKT_DATA_MEASURE "../RIM_data/wiki_talk/meas.csv"
#define WKT_PATH "../Graph_Data_Storage/wiki_talk.csv"
#define WKT_DATA_PATH "../Graph_Data_Storage/wiki_talk-data.csv"
#define WKT_SEED_PATH "../RIM_res/res_wiki_talk.csv"

#define MAX_WHILE 100 

using namespace std;

struct edge{
    unsigned int src;
    unsigned int dst;
};

void readData(string filename, edge* edge_list);

void get_graph_info(string path, unsigned int* nodes, unsigned int* edges);

void genCSC(edge* edge_list, unsigned int* succ, unsigned int* csc, unsigned int node_size, unsigned int edge_size);


template <typename idx_t>
void genCSR(edge* edge_list, idx_t* src, idx_t* succ, unsigned int node_size, unsigned int edge_size){
    for(int i=0; i<edge_size;i++){
        src[edge_list[i].src]++;
        succ[i]=edge_list[i].dst;
    }
    //Now, we need to prefix sum the src_ptr
    idx_t* src_temp = new idx_t[node_size+1]{0};
    for(int i=1; i<=node_size;i++){
        src_temp[i]=src_temp[i-1]+src[i-1];
    }
    copy(src_temp, src_temp+node_size+1, src);
    delete[] src_temp;
}

void GenAdj(edge* edge_list, float* adj, unsigned int node_size, unsigned int edge_size);

void h_MatVecMult(float* h_A, float* h_x, float* h_y, unsigned int node_size);

void Normalize_L2(float* h_x, unsigned int node_size);

void Export_Seed_Set_to_CSV(unsigned int* seed_set, unsigned int seed_size, string path);


__host__ void Save_Data(string file, float time, float damping_factor, float threshold,unsigned int epoch);


//CUDA

__host__ void  RIM_rand_Ver1(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file);

__host__ void  RIM_rand_Ver2(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file);


__host__ void  RIM_rand_Ver3_PR(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, edge* edge_list, string file, string pr_file);

template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult(IndexType* csc, IndexType* succ, float* values, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        for(IndexType i = start; i < end; i++){
            sum += values[i]*vec[succ[i]];
        }
        result[t] = sum;
    }
}

__global__ void Float_VectAdd(float* vec1, float* vec2, unsigned int size);

__global__ void Init_Random(float* vec, float* rand_init, unsigned int size, unsigned int k);

__host__ void Verify(float* gpu_vec, float* cpu_vec, unsigned int size);

__host__ void CheckSparseMatVec(unsigned int* csc, unsigned int* succ,edge* edge_list, unsigned int node_size, unsigned int edge_size);

__host__ void PageRank(float* pr_vector, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time);

__host__ void PageRank_Sparse(float* pr_vector, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time, string pr_file);

__global__ void Init_Pr(float* pr_vector, unsigned int node_size);

__global__ void Gen_P_Mem_eff(float* weight_P, unsigned int* src, unsigned int* succ, unsigned int node_size, float* damp);

__global__ void Init_P(float* P, unsigned int node_size, float* damp);

__host__ void  RIM_rand_Ver4_Greedy(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver5_Sig(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver6_Tanh(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver7_PR_Rand(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__global__ void Zero_Rows(float* values, unsigned int* csc, unsigned int* succ, unsigned int* idx, unsigned int node_size, unsigned int num_cancel);

__global__ void Init_P_Sparse(float* weight_P,const int* src,const int* succ, unsigned int node_size, float* damp);

__host__ void Verify_Pr(float* sparse_vec, float* full_vec, unsigned int node_size);

__host__ void Gen_Pr_Sprs(unsigned int* csc, unsigned int* succ, float* weight_P, unsigned int node_size, unsigned int edge_size, float damp, string file);


//Device Functions

__device__ float eval_values(float rand_num, float val,float threshold);

__device__ float eval_values_v2(float rand_num, float val,float threshold);