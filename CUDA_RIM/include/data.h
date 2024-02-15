#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>   
#include <algorithm>
#include <cstdlib>
#include <ctime>
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
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <cusolverDn.h>
#include "GPUErrors.h"

#define TPB 256
#define K 100

#define NUMSTRM 10

#define HOMO_PATH "../Graph_Data_Storage/homo.csv"
#define HOMO_CSC_PATH "../Graph_Data_Storage/homo_csc.csv"
#define HOMO_DATA_PATH "../Graph_Data_Storage/homo_info.csv"
#define SEED_PATH "../RIM_res/res_4000.csv"
#define ARVIX_PATH "../Graph_Data_Storage/ca-GrQc.csv"
#define ARVIX_CSC_PATH "../Graph_Data_Storage/ca-GrQc_csc.csv"
#define ARVIX_DATA_PATH "../Graph_Data_Storage/ca-GrQc-data.csv"
#define ARVIX_SEED_PATH "../RIM_res/res_arvix.csv"
#define WIKI_VOTE_PATH "../Graph_Data_Storage/wikivote.csv"
#define WIKI_VOTE_CSC_PATH "../Graph_Data_Storage/wikivote_csc.csv"
#define WIKI_VOTE_DATA_PATH "../Graph_Data_Storage/wikivote_data.csv"
#define WIKI_VOTE_SEED_PATH "../RIM_res/res_wiki.csv"
#define EP_PATH "../Graph_Data_Storage/epinions.csv"
#define EP_CSC_PATH "../Graph_Data_Storage/epinions_csc.csv"
#define EP_DATA_PATH "../Graph_Data_Storage/epinions_data.csv"
#define EP_SEED_PATH "../RIM_res/res_ep.csv"
#define HEPTH_PATH "../Graph_Data_Storage/ca-HepTh.csv"
#define HEPTH_CSC_PATH "../Graph_Data_Storage/ca-HepTh_csc.csv"
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
#define HEPTH_DATA_MEASURE_RPR "../RIM_data/HepTh/meas_7.csv"
#define WIKI_VOTE_DATA_MEASURE_RPR "../RIM_data/wiki-vote/meas_7.csv"
#define ARVIX_DATA_MEASURE_RPR "../RIM_data/arvix/meas_7.csv"
#define HOMO_DATA_MEASURE_RPR "../RIM_data/syn/meas_7.csv"
#define EP_DATA_MEASURE_RPR "../RIM_data/epinions/meas_7.csv"

#define HEPTH_DATA_MEASURE_BFS "../RIM_data/HepTh/meas_8.csv"
#define WIKI_VOTE_DATA_MEASURE_BFS "../RIM_data/wiki-vote/meas_8.csv"
#define ARVIX_DATA_MEASURE_BFS "../RIM_data/arvix/meas_8.csv"
#define HOMO_DATA_MEASURE_BFS "../RIM_data/syn/meas_8.csv"
#define EP_DATA_MEASURE_BFS "../RIM_data/epinions/meas_8.csv"
#define HEPTH_PR "../Graph_Data_Storage/hepth_pr.csv"
#define WIKI_VOTE_PR "../Graph_Data_Storage/wikivote_pr.csv"
#define ARVIX_PR "../Graph_Data_Storage/arxiv_pr.csv"
#define HOMO_PR "../Graph_Data_Storage/homo_pr.csv"
#define EP_PR "../Graph_Data_Storage/epinions_pr.csv"
#define AM_PR "../Graph_Data_Storage/amazon_pr.csv"
#define AM_DATA_MEASURE_GREEDY "../RIM_data/amazon/meas_4.csv"
#define AM_DATA_MEASURE_SIG "../RIM_data/amazon/meas_5.csv"
#define AM_DATA_MEASURE_TANH "../RIM_data/amazon/meas_6.csv"
#define AM_DATA_MEASURE_RPR "../RIM_data/amazon/meas_7.csv"
#define AM_DATA_MEASURE_BFS "../RIM_data/amazon/meas_8.csv"
#define AM_DATA_MEASURE_PR "../RIM_data/amazon/meas_3.csv"
#define AM_DATA_MEASURE_2 "../RIM_data/amazon/meas_2.csv"
#define AM_DATA_MEASURE "../RIM_data/amazon/meas.csv"
#define AM_PATH "../Graph_Data_Storage/amazon.csv"
#define AM_CSC_PATH "../Graph_Data_Storage/amazon_csc.csv"
#define AM_DATA_PATH "../Graph_Data_Storage/amazon-data.csv"
#define AM_SEED_PATH "../RIM_res/res_amazon.csv"
#define ND_PR "../Graph_Data_Storage/nd_pr.csv"
#define ND_DATA_MEASURE_GREEDY "../RIM_data/nd/meas_4.csv"
#define ND_DATA_MEASURE_SIG "../RIM_data/nd/meas_5.csv"
#define ND_DATA_MEASURE_TANH "../RIM_data/nd/meas_6.csv"
#define ND_DATA_MEASURE_RPR "../RIM_data/nd/meas_7.csv"
#define ND_DATA_MEASURE_BFS "../RIM_data/nd/meas_8.csv"
#define ND_DATA_MEASURE_PR "../RIM_data/nd/meas_3.csv"
#define ND_DATA_MEASURE_2 "../RIM_data/nd/meas_2.csv"
#define ND_DATA_MEASURE "../RIM_data/nd/meas.csv"
#define ND_PATH "../Graph_Data_Storage/nd.csv"
#define ND_CSC_PATH "../Graph_Data_Storage/nd_csc.csv"
#define ND_DATA_PATH "../Graph_Data_Storage/nd-data.csv"
#define ND_SEED_PATH "../RIM_res/res_nd.csv"
#define BRK_PR "../Graph_Data_Storage/berk_pr.csv"
#define BRK_DATA_MEASURE_GREEDY "../RIM_data/berk/meas_4.csv"
#define BRK_DATA_MEASURE_SIG "../RIM_data/berk/meas_5.csv"
#define BRK_DATA_MEASURE_TANH "../RIM_data/berk/meas_6.csv"
#define BRK_DATA_MEASURE_RPR "../RIM_data/berk/meas_7.csv"
#define BRK_DATA_MEASURE_BFS "../RIM_data/berk/meas_8.csv"
#define BRK_DATA_MEASURE_PR "../RIM_data/berk/meas_3.csv"
#define BRK_DATA_MEASURE_2 "../RIM_data/berk/meas_2.csv"
#define BRK_DATA_MEASURE "../RIM_data/berk/meas.csv"
#define BRK_PATH "../Graph_Data_Storage/berk.csv"
#define BRK_CSC_PATH "../Graph_Data_Storage/berk_csc.csv"
#define BRK_DATA_PATH "../Graph_Data_Storage/berk-data.csv"
#define BRK_SEED_PATH "../RIM_res/res_berk.csv"
#define GGL_PR "../Graph_Data_Storage/google_pr.csv"
#define GGL_DATA_MEASURE_GREEDY "../RIM_data/google/meas_4.csv"
#define GGL_DATA_MEASURE_SIG "../RIM_data/google/meas_5.csv"
#define GGL_DATA_MEASURE_TANH "../RIM_data/google/meas_6.csv"
#define GGL_DATA_MEASURE_RPR "../RIM_data/google/meas_7.csv"
#define GGL_DATA_MEASURE_BFS "../RIM_data/google/meas_8.csv"
#define GGL_DATA_MEASURE_PR "../RIM_data/google/meas_3.csv"
#define GGL_DATA_MEASURE_2 "../RIM_data/google/meas_2.csv"
#define GGL_DATA_MEASURE "../RIM_data/google/meas.csv"
#define GGL_PATH "../Graph_Data_Storage/google.csv"
#define GGL_CSC_PATH "../Graph_Data_Storage/google_csc.csv"
#define GGL_DATA_PATH "../Graph_Data_Storage/google-data.csv"
#define GGL_SEED_PATH "../RIM_res/res_google.csv"
#define WKT_PR "../Graph_Data_Storage/wiki_talk_pr.csv"
#define WKT_DATA_MEASURE_GREEDY "../RIM_data/wiki_talk/meas_4.csv"
#define WKT_DATA_MEASURE_SIG "../RIM_data/wiki_talk/meas_5.csv"
#define WKT_DATA_MEASURE_TANH "../RIM_data/wiki_talk/meas_6.csv"
#define WKT_DATA_MEASURE_RPR "../RIM_data/wiki_talk/meas_7.csv"
#define WKT_DATA_MEASURE_BFS "../RIM_data/wiki_talk/meas_8.csv"
#define WKT_DATA_MEASURE_PR "../RIM_data/wiki_talk/meas_3.csv"
#define WKT_DATA_MEASURE_2 "../RIM_data/wiki_talk/meas_2.csv"
#define WKT_DATA_MEASURE "../RIM_data/wiki_talk/meas.csv"
#define WKT_PATH "../Graph_Data_Storage/wiki_talk.csv"
#define WKT_CSC_PATH "../Graph_Data_Storage/wiki_talk_csc.csv"
#define WKT_DATA_PATH "../Graph_Data_Storage/wiki_talk-data.csv"
#define WKT_SEED_PATH "../RIM_res/res_wiki_talk.csv"

#define COUNT_PATH "../RIM_res/count.csv"
#define SCORE_PATH "../RIM_res/score.csv"


#define MAX_WHILE 200

using namespace std;

struct edge{
    unsigned int src;
    unsigned int dst;
};

struct ValueTuple {
    unsigned int idx;
    unsigned int count;
};

void readData(string filename, edge* edge_list);

void get_graph_info(string path, unsigned int* nodes, unsigned int* edges);




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

template <typename idx_t>
void genCSC(edge* edge_list, idx_t* csc, idx_t* succ, unsigned int node_size, unsigned int edge_size){
    unsigned int* csc_temp = new unsigned int[node_size+1]{0};
    for(unsigned int i = 0; i < edge_size; i++){
        csc_temp[edge_list[i].dst]++;
        succ[i] = edge_list[i].src;
    }
    csc[0] = 0;
    for(unsigned int i = 1; i<=node_size; i++){
        csc[i] = csc[i-1] + csc_temp[i-1];
    }
    delete[] csc_temp;
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


template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult_Mart_BFS(IndexType* csc, IndexType* succ, float* values, float* vec, float* result, unsigned int* visited, float threshold,unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        for(IndexType i = start; i < end; i++){
            sum += values[i]*vec[succ[i]]*(visited[succ[i]]);
            // if(visited[succ[i]]==1){
            //     printf("Vec: %f\n",vec[succ[i]]);
            //     printf("values: %f\n",values[i]);
            // }
        }
        sum*= exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
        if(visited[t]==0){
            result[t] += sum;
            if(sum>0){
                atomicExch(&visited[t],1);
            }
        }
    }
}

template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult_Mart_BFS_v2(IndexType* csc, IndexType* succ, float* values, float* vec, float* result, unsigned int* visited, float* penalty, float threshold,unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        for(IndexType i = start; i < end; i++){
            sum += values[i]*vec[succ[i]];
        }
        sum*= exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
        if(visited[t]==0 && sum>0){
            result[t] = sum*penalty[t];
            visited[t] =1;
        }
    }
}

template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult_Mart_BFS_v3(IndexType* csc, IndexType* succ, float* values, float* vec, float* result, unsigned int* visited, float threshold,unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        for(IndexType i = start; i < end; i++){
            sum += values[i]*vec[succ[i]];
        }
        sum*= exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
        if(visited[t]==0){
            result[t] += sum;
            atomicExch(&visited[t],1);
        }
        if(visited[t]==1){
            result[t] = 0;
            atomicExch(&visited[t],0);
        }
    }
}

__global__ void Zero_Cols_Max_Idx(float* values, unsigned int* csc, unsigned int* succ, unsigned int* idx, unsigned int node_size, unsigned int edge_size, unsigned int num_cancel);

__device__ float sigmoid(float x);

__device__ float tanh_dev(float x);

__device__ float swish(float x);


template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult_BFS(IndexType* csc, IndexType* succ, IndexType* visited, float* values, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        if(visited[t]){
            IndexType start = csc[t];
            IndexType end = csc[t+1];
            float sum = 0.0f;
            for(IndexType i = start; i < end; i++){
                //if it is visted, set it to 0, otherwise, set it to 1
                sum += values[i]*vec[succ[i]]*(visited[succ[i]]);
                visited[succ[i]]=0;
            }
            result[t] = sum;
        }
        else{
            result[t] = 0.0f;
        }
    }
}


template <typename IndexType>
__global__ void sparseCSRMat_SparseDiagMat_Mult_BFS(IndexType* csc, IndexType* succ, float* values, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        for(IndexType i = start; i < end; i++){
            sum += values[i] * vec[t] * vec[succ[i]];
        }
        result[t] = sum;
    }
}


template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult_BFS_Sig(IndexType* csc, IndexType* succ, IndexType* visited, float* values, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        float temp = 0.0f;
        for(IndexType i = start; i < end; i++){
            //if it is visted, set it to 0, otherwise, set it to 1
            temp = values[i]*vec[succ[i]]*(visited[succ[i]]);
            sum += sigmoid(temp);
            visited[succ[i]]=visited[succ[i]]==1?(0):(visited[succ[i]]);
        }
        result[t] = sum;
    }
}


template <typename IndexType>
__global__ void sparseCSRMat_Vec_Mult_BFS_Tanh(IndexType* csc, IndexType* succ, IndexType* visited, float* values, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        IndexType start = csc[t];
        IndexType end = csc[t+1];
        float sum = 0.0f;
        float temp = 0.0f;
        for(IndexType i = start; i < end; i++){
            //if it is visted, set it to 0, otherwise, set it to 1
            temp = values[i]*vec[succ[i]]*(visited[succ[i]]);
            sum += tanh_dev(temp);
            visited[succ[i]]=visited[succ[i]]==1?(0):(visited[succ[i]]);
        }
        result[t] = sum;
    }
}

__global__ void Condense_Score(float* fin, float* inter, unsigned int node_size, unsigned int num_strm);

template <typename T>
__global__ void Copy(T* d_res, T* d_vec, unsigned int node_size);

__global__ void Float_VectAdd(float* vec1, float* vec2, unsigned int size);

__global__ void Init_Random(float* vec, float* rand_init, unsigned int size, unsigned int k);

__host__ void Verify(float* gpu_vec, float* cpu_vec, unsigned int size);

__host__ void CheckSparseMatVec(unsigned int* csc, unsigned int* succ,edge* edge_list, unsigned int node_size, unsigned int edge_size);

__host__ void PageRank(float* pr_vector, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time);

__host__ void PageRank_Sparse(float* pr_vector, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time, string pr_file);

__host__ void Export_Counts(string file, unsigned int* count,unsigned int* idx, unsigned int node_size);

__global__ void Init_Pr(float* pr_vector, unsigned int node_size);

__global__ void Gen_P_Mem_eff(float* weight_P, unsigned int* src, unsigned int* succ, unsigned int node_size, float* damp);

__global__ void Init_P(float* P, unsigned int node_size, float* damp);

__global__ void Calc_Penalty(float* d_res, float* d_penality, unsigned int node_size);

__global__ void Float_VectAdd_Cap(float* vec1, float* vec2, unsigned int* idx, unsigned int size, unsigned int idx_size);

__host__ void  RIM_rand_Ver4_Greedy(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver5_Sig(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver6_Tanh(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver7_PR_Rand(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Ver9_BFS(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file);

__host__ void  RIM_rand_Mart_BFS(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file);

__global__ void Zero_Rows(float* values, unsigned int* csc, unsigned int* succ, unsigned int* idx, unsigned int node_size, unsigned int num_cancel);

__global__ void Init_P_Sparse(float* weight_P,const int* src,const int* succ, unsigned int node_size, float* damp);

__host__ void Verify_Pr(float* sparse_vec, float* full_vec, unsigned int node_size);

__host__ void Gen_Pr_Sprs(unsigned int* csc, unsigned int* succ, float* weight_P, unsigned int node_size, unsigned int edge_size, float damp, string file);

__global__ void Int_PointAdd(int* vec1, int* vec2, unsigned int size);

__global__ void Fill_Diag(float* A, float* diag, unsigned int node_size, unsigned int size);

__global__ void Prob_BFS_Score_Kernel(unsigned int* d_csc, unsigned int* d_succ, unsigned int node_size, unsigned int edge_size, float* d_score, unsigned int* d_visited,
unsigned int* frontier, unsigned int* next_frontier, float threshold, int level);

__host__ void  RIM_rand_Mart_BFS_v2(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file);


__host__ void  RIM_rand_Mart_BFS_v3(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file);


__host__ void Prob_BFS_Score(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file);

__global__ void Zero_Rows_Max_Idx(float* values, unsigned int* csc, unsigned int* succ, unsigned int* idx, unsigned int node_size, unsigned int num_cancel);
//Device Functions

__host__ void Export_Scores(string file, float* scores, unsigned int* idx, unsigned int node_size);

__global__ void Transform_Bool(float* d_res, unsigned int* d_vec, unsigned int node_size);

__device__ float eval_values(float rand_num, float val,float threshold);

__device__ float eval_values_v2(float rand_num, float val,float threshold);


void Split_Tuple_Count(unsigned int* count, unsigned int* idx, ValueTuple* list, unsigned int node_size);

void Make_Tuple_Count(unsigned int* count, unsigned int* idx, ValueTuple* list, unsigned int node_size);



