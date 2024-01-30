#include "../include/data.h"


__host__ void Prob_BFS_Score(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file, string pr_file){
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_score;
    float* d_share_score;
    unsigned int* d_visited;
    unsigned int* d_seed_set;
    float* rand_numbers;
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int) * (node_size + 1)))){
        cout << "cudaMalloc d_csc failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int) * edge_size))){
        cout << "cudaMalloc d_succ failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_score, sizeof(float) * node_size * NUMSTRM))){
        cout << "cudaMalloc d_score failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_visited, sizeof(unsigned int) * node_size * NUMSTRM))){
        cout << "cudaMalloc d_visited failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_share_score, sizeof(float) * node_size))){
        cout << "cudaMalloc d_share_score failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * NUMSTRM))){
        cout << "cudaMalloc rand_numbers failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int) * (node_size + 1), cudaMemcpyHostToDevice))){
        cout << "cudaMemcpy d_csc failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int) * edge_size, cudaMemcpyHostToDevice))){
        cout << "cudaMemcpy d_succ failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_seed_set, sizeof(unsigned int) * K))){
        cout << "cudaMalloc d_seed_set failed" << endl;
        exit(0);
    }
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    CreateStreams(NUMSTRM, streams);
    unsigned int max_blocks = Max_Blocks(TPB,NUMSTRM);
    unsigned int blocks = node_size/TPB+1;
    if(blocks > max_blocks){
        blocks = max_blocks;
    }
    thrust::fill(thrust::device, d_score, d_score + node_size * NUMSTRM, 0);
    thrust::fill(thrust::device, d_visited, d_visited + node_size * NUMSTRM, 0);
    thrust::fill(thrust::device, d_share_score, d_share_score + node_size, 0);
    thrust::fill(thrust::device, d_seed_set, d_seed_set + K, 0);
    thrust::fill(thrust::device, rand_numbers, rand_numbers + NUMSTRM, 0);
    unsigned int iter = 5;
    for(int i = 0; i < iter; i++){
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_numbers, edge_size);
        curandDestroyGenerator(gen);
        cout << "iter: " << i << endl;
        for(int j = 0; j<NUMSTRM;j++){
            Prob_BFS_Score_Kernel<<<blocks, TPB, 0, streams[j]>>>(d_csc, d_succ, node_size, edge_size, d_score, d_visited, threshold);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[j]))){
                cout << "cudaStreamSynchronize failed" << endl;
                exit(0);
            }
        }

    }

}

__global__ void Prob_BFS_Score_Kernel(unsigned int* d_csc, unsigned int* d_succ, unsigned int node_size, unsigned int edge_size, float* d_score, unsigned int* d_visited, float threshold){

}