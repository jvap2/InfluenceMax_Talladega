#include "../include/data.h"


__host__ void Prob_BFS_Score(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file){
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_score;
    float* d_share_score;
    unsigned int* d_visited;
    unsigned int* d_seed_set;
    unsigned int* d_frontier;
    unsigned int* d_next_frontier;
    float* rand_numbers;
    unsigned int* rand_idx;
    float* res = new float[NUMSTRM];
    thrust::fill(thrust::host, res, res + NUMSTRM, 0);
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int) * (node_size + 1)))){
        std::cout << "cudaMalloc d_csc failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int) * edge_size))){
        std::cout << "cudaMalloc d_succ failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_score, sizeof(float) * node_size * NUMSTRM))){
        std::cout << "cudaMalloc d_score failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_visited, sizeof(unsigned int) * node_size * NUMSTRM))){
        std::cout << "cudaMalloc d_visited failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_share_score, sizeof(float) * node_size))){
        std::cout << "cudaMalloc d_share_score failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * NUMSTRM))){
        std::cout << "cudaMalloc rand_numbers failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_frontier, sizeof(unsigned int) * node_size * NUMSTRM))){
        std::cout << "cudaMalloc d_frontier failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_next_frontier, sizeof(unsigned int) * node_size * NUMSTRM))){
        std::cout << "cudaMalloc d_next_frontier failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&rand_idx, sizeof(unsigned int) * node_size))){
        std::cout << "cudaMalloc rand_idx failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int) * (node_size + 1), cudaMemcpyHostToDevice))){
        std::cout << "cudaMemcpy d_csc failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int) * edge_size, cudaMemcpyHostToDevice))){
        std::cout << "cudaMemcpy d_succ failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_seed_set, sizeof(unsigned int) * K))){
        std::cout << "cudaMalloc d_seed_set failed" << endl;
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
    unsigned int total_visited = 0; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++){
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_numbers, NUMSTRM+1);
        curandDestroyGenerator(gen);
        thrust::transform(thrust::device, rand_numbers, rand_numbers + NUMSTRM, rand_numbers, [=] __device__ (float rand_num) { return rand_num*node_size; });
        for(int j = 0; j<NUMSTRM;j++){
            unsigned int* frontier_j = d_frontier + j * node_size;
            thrust::transform(thrust::device, rand_numbers+j, rand_numbers + j+1, frontier_j, [=] __device__ (float rand_num) {
                int index = static_cast<int>(rand_num);
                if (index >= 0 && index < node_size) {
                    return 1;
                } else {
                    return 0;
                }
            });
        }
        std::cout << "iter: " << i << endl;
        unsigned int level = 1;
        do{
            for(int j = 0; j<NUMSTRM;j++){
                unsigned int* d_visited_j = d_visited + j * node_size;
                float* d_score_j = d_score + j * node_size;
                unsigned int* frontier_j = d_frontier + j * node_size;
                unsigned int* next_j = d_next_frontier + j * node_size;
                Prob_BFS_Score_Kernel<<<blocks, TPB, 0, streams[j]>>>(d_csc, d_succ, node_size, edge_size, d_score_j, d_visited_j, frontier_j, next_j, threshold, level);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[j]))){
                    std::cout << "cudaStreamSynchronize failed" << endl;
                    exit(0);
                }
                thrust::copy(thrust::device.on(streams[j]), frontier_j, frontier_j + node_size, next_j);
                thrust::fill(thrust::device.on(streams[j]), next_j, next_j + node_size, 0);
                res[j]=thrust::reduce(thrust::device.on(streams[j]), d_visited_j, d_visited_j + node_size, 0);
            }
            total_visited = thrust::reduce(thrust::host, res, res + NUMSTRM, 0);
        }while(total_visited < NUMSTRM*node_size);
        //Collect the sum of the scores and place into shared score
        for(int k = 0; k<NUMSTRM;k++){
            float* d_score_k = d_score + k * node_size;
            // unsigned int strm_no = NUMSTRM;
            // thrust::transform(thrust::device.on(streams[k]), d_score_k, d_score_k + node_size, d_score_k, [strm_no] __device__ (float score) { return score / strm_no; });
            thrust::transform(thrust::device.on(streams[k]), d_score_k, d_score_k + node_size, d_score_k, [=] __device__ (float score) { return exp(score); });
            thrust::transform(thrust::device.on(streams[k]), d_score_k, d_score_k + node_size, d_share_score, d_share_score, thrust::plus<float>());
        }

    }
    thrust::sequence(thrust::device, rand_idx, rand_idx + node_size, 0);
    // Sort d_share_score in descending order
    thrust::sort_by_key(thrust::device, d_share_score, d_share_score+node_size, rand_idx, thrust::greater<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << "ms" << endl;
    unsigned int* h_rand_idx = new unsigned int[node_size];
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
    }
    delete[] h_rand_idx;
    if(!HandleCUDAError(cudaFree(d_csc))){
        std::cout << "cudaFree d_csc failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_succ))){
        std::cout << "cudaFree d_succ failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_score))){
        std::cout << "cudaFree d_score failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_visited))){
        std::cout << "cudaFree d_visited failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_share_score))){
        std::cout << "cudaFree d_share_score failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_seed_set))){
        std::cout << "cudaFree d_seed_set failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_frontier))){
        std::cout << "cudaFree d_frontier failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(d_next_frontier))){
        std::cout << "cudaFree d_next_frontier failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(rand_numbers))){
        std::cout << "cudaFree rand_numbers failed" << endl;
        exit(0);
    }
    if(!HandleCUDAError(cudaFree(rand_idx))){
        std::cout << "cudaFree rand_idx failed" << endl;
        exit(0);
    }
    free(streams);
    delete[] res;
    if(!HandleCUDAError(cudaDeviceReset())){
        std::cout << "cudaDeviceReset failed" << endl;
        exit(0);
    }

}

__global__ void Prob_BFS_Score_Kernel(unsigned int* d_csc, unsigned int* d_succ, unsigned int node_size, unsigned int edge_size, float* d_score, unsigned int* d_visited,
unsigned int* frontier, unsigned int* next_frontier, float threshold, int level)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < node_size; i += blockDim.x * gridDim.x){
        if(frontier[i]==1){
            //This means it is in this current frontier
            //We need to go to the next frontier
            frontier[i] = 0;
            unsigned int start = d_csc[i];
            unsigned int end = d_csc[i+1];
            for(int j = start; j < end; j++){
                unsigned int neighbor = d_succ[j];
                if(d_visited[neighbor]==0){
                    //This means it has not been visited
                    //We need to add it to the next frontier
                    d_visited[neighbor] = 1;
                    next_frontier[neighbor] = 1;
                    if(level <=1)
                        d_score[neighbor] += score_function(threshold);
                    else
                        d_score[neighbor] += score_function(threshold)*d_score[i];
                }
            }
        }
    }

}


__device__ float score_function(float threshold){
    return exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
}