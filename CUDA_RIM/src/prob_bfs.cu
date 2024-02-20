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
    thrust::fill(thrust::device, d_share_score, d_share_score + node_size, 0);
    thrust::fill(thrust::device, d_seed_set, d_seed_set + K, 0);
    unsigned int iter = 5;
    unsigned int total_visited = 0; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < iter; i++){
        thrust::fill(thrust::device, d_score, d_score + node_size * NUMSTRM, 0);
        thrust::fill(thrust::device, d_visited, d_visited + node_size * NUMSTRM, 0);
        thrust::fill(thrust::device, rand_numbers, rand_numbers + NUMSTRM, 0);
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
                res[j]=thrust::reduce(thrust::device.on(streams[j]), d_visited_j, d_visited_j + node_size, 0);
                // cout<<"res[j]: "<<res[j]<<endl;
                // cout<<"Node size: "<<node_size<<endl;   
            }
            thrust::copy(thrust::device, d_next_frontier, d_next_frontier + node_size * NUMSTRM, d_frontier);
            thrust::fill(thrust::device, d_next_frontier, d_next_frontier + node_size * NUMSTRM, 0);
            total_visited = thrust::reduce(thrust::host, res, res + NUMSTRM, 0);
            level++;
        }while(total_visited < NUMSTRM*node_size && level<=10);
        //Collect the sum of the scores and place into shared score
        for(int k = 0; k<NUMSTRM;k++){
            float* d_score_k = d_score + k * node_size;
            // thrust::transform(thrust::device.on(streams[k]), d_score_k, d_score_k + node_size, d_score_k, [=] __device__ (float score) { return exp(score); });
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
    float* debug = new float[node_size];
    if(!HandleCUDAError(cudaMemcpy(debug, d_share_score, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying d_share_score to host"<<endl;
    }
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
    for(int i = 0; i < K; i++){
        cout<<seed_set[i]<<endl;
        cout<<debug[i]<<endl;
    }
    delete[] debug;
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
                    if(level <=1){
                        d_score[neighbor] += exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
                    }
                    else
                        d_score[neighbor] += exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1))*d_score[i];
                }
            }
        }
    }
}



__host__ void  RIM_rand_Mart_BFS(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file){
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int num_walker = 1;
    unsigned int epochs=30;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* d_track_val;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* temp_sum = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
    unsigned int* d_check;
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*(node_size+1)))){
        std::cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size)))){
        std::cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_track_val, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_check, sizeof(unsigned int)*NUMSTRM*node_size))){
        std::cout<<"Error allocating memory for d_check"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_vec, vec, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying vec to device"<<endl;
    }
    delete[] vec;
    if(!HandleCUDAError(cudaMemcpy(d_res, res, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying res to device"<<endl;
    }
    delete[] res;

    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size*NUMSTRM)))){
        std::cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_values, values, sizeof(float)*edge_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying values to device"<<endl;
    }
    // delete[] values;


    float* rand_init;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_init, NUMSTRM*num_walker*sizeof(float)))){
        std::cout<<"Error allocating memory for rand_frog"<<endl;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /*Now, we have the random numbers generated*/
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    if(!HandleCUDAError(cudaMalloc((void**)&rand_vec_init, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for rand_vec_init"<<endl;
    } 
    if(!HandleCUDAError(cudaMemcpy(rand_vec_init, h_rand_vec_init, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying h_rand_vec_init to device"<<endl;
    }

    float* store_stream_res;
    float* h_store_stream_res = new float[node_size*NUMSTRM];
    thrust::fill(h_store_stream_res, h_store_stream_res+node_size*NUMSTRM, 0.0f);
    if(!HandleCUDAError(cudaMalloc((void**)&store_stream_res, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for store_stream_res"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(store_stream_res, h_store_stream_res, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying h_store_stream_res to device"<<endl;
    }
    delete[] h_store_stream_res;
    float* rand_numbers;
    if (!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * edge_size*NUMSTRM))) {
        std::cout << "Error allocating memory for rand_numbers" << endl;
    }
    printCudaMemoryUsage();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float tol_thresh=1;
    epochs = 30*(K/NUMSTRM+1);
    for(int k = 0; k < epochs; k++){
        // std::cout<<"Epoch "<<k<<endl;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        thrust::fill(sum, sum+NUMSTRM, 0.0f);
        thrust::fill(temp_sum, temp_sum+NUMSTRM, 0.0f);
        int while_count = 0;
        while_count=0;
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_init, num_walker*NUMSTRM);
        for(int i = 0; i < NUMSTRM; i++){
            //Initialize the random vector
            float* rand_init_i = rand_init + i*num_walker;
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            unsigned int* d_check_i = d_check + i*node_size;
            thrust::fill(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, 0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_check_i, d_check_i+node_size, 0);
            thrust::fill(thrust::device.on(streams[i]),d_res+i*node_size, d_res+i*node_size+node_size, 0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_track_val+i*node_size, d_track_val+i*node_size+node_size, 0.0f);
            Init_Random<<<blocks_per_stream, TPB,0,streams[i]>>>(rand_vec_init_i, rand_init_i, node_size, num_walker);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device at Init Random for Stream "<<i<<endl;
            }
            float* rand_numbers_i = rand_numbers + i*edge_size;
            float* d_values_i = d_values + i*edge_size;
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            srand(time(0));
            int rand_seed = rand();
            curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            curandGenerateUniform(gen, rand_numbers_i, edge_size);
            curandDestroyGenerator(gen);
            thrust::fill(thrust::device.on(streams[i]), d_values_i, d_values_i+edge_size, 1.0f);
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
            // cout<<"While count: "<<while_count<<endl;
            float level_thresh = exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
            for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > 0){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    float* d_track_i = d_track_val + i*node_size;
                    unsigned int* d_check_i = d_check + i*node_size;
                    sparseCSRMat_Vec_Mult_Mart_BFS<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i,d_check_i, threshold, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                    }
                    // thrust::transform(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, d_res_i, [=] __device__ (float x) { return x*level_thresh; });
                    Copy<float><<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i, rand_vec_init_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at Copy for stream "<<i<<endl;
                    }
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_track_i, rand_vec_init_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at Float_VectAdd for stream "<<i<<endl;
                    }
                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), d_check_i, d_check_i+node_size);
                    tol[i] = sum[i]-temp_sum[i];
                    temp_sum[i] = sum[i];
                    // cout<<"Tol: "<<tol[i]<<endl;
                    // cout<<"Level Thresh: "<<level_thresh<<endl;
                    // cout<<"sum[i]: "<<sum[i]<<endl; 
                    thrust::fill(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, 0.0f);
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > 0){
                    check = true;
                }
            }
        }
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = d_track_val + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            //Take the softmax of rand_vec_init_i
            sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
            float temp = sum[i];
            // cout<<"Sum: "<<sum[i]<<endl;
            if(sum[i]>0){
                thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                }
            }
            thrust::fill(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, 0);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    curandDestroyGenerator(gen);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file,blocks_per_stream, milliseconds, damping_factor, threshold, epochs);
    if(!HandleCUDAError(cudaFree(d_csc))){
        std::cout<<"Error freeing d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_succ))){
        std::cout<<"Error freeing d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_vec))){
        std::cout<<"Error freeing d_vec"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_res))){
        std::cout<<"Error freeing d_res"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_values))){
        std::cout<<"Error freeing d_values"<<endl;
    }
    unsigned int* rand_idx;
    unsigned int* h_rand_idx = new unsigned int[node_size];
    thrust::fill(h_rand_idx, h_rand_idx+node_size, 0);
    if(!HandleCUDAError(cudaMalloc((void**)&rand_idx, sizeof(unsigned int)*node_size))){
        std::cout<<"Error allocating memory for rand_idx"<<endl;
    }
    thrust::sequence(thrust::device, rand_idx, rand_idx+node_size);
    //Take the sum of the vectors and then sort them
    for(int i = 1; i<NUMSTRM;i++){
        thrust::transform(thrust::device, store_stream_res, store_stream_res+node_size, store_stream_res+i*node_size, store_stream_res, thrust::plus<float>());
    }
    thrust::sort_by_key(thrust::device, store_stream_res, store_stream_res+node_size, rand_idx, thrust::greater<float>());
    //Get the top k indexes
    float* h_store_stream_res_fin = new float[node_size*NUMSTRM];
    if(!HandleCUDAError(cudaMemcpy(h_store_stream_res_fin, store_stream_res, sizeof(float)*node_size*NUMSTRM, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying store_stream_res to host"<<endl;
    }
    
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
        cout<<seed_set[i]<<endl;
        cout<<h_store_stream_res_fin[i]<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_idx))){
        std::cout<<"Error freeing rand_idx"<<endl;
    }
    delete[] h_rand_idx;

    for(int i = 0; i<NUMSTRM;i++){
        if(!HandleCUDAError(cudaStreamDestroy(streams[i]))){
            std::cout<<"Error destroying stream number "<<i<<endl;
        }
    }
    if(!HandleCUDAError(cudaFree(store_stream_res))){
        std::cout<<"Error freeing store_stream_res"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_numbers))){
        std::cout<<"Error freeing rand_numbers"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_vec_init))){
        std::cout<<"Error freeing rand_vec_init"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_init))){
        std::cout<<"Error freeing rand_init"<<endl;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        std::cout<<"Error resetting device"<<endl;
    }
    delete[] h_rand_vec_init;
    delete[] l2_norm_rand_vec_init;
    delete[] l2_norm_d_res;
    delete[] sum;
    delete[] tol;
    delete[] values;

}


__host__ void  RIM_rand_Mart_BFS_v2(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file){
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int num_walker = 1;
    unsigned int epochs;
    unsigned int* d_csc;
    unsigned int* d_succ;
    unsigned int* count = new unsigned int[node_size];
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* d_track_val;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* temp_sum = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    float* d_penality;
    float* d_values_temp;
    unsigned int* max_index = new unsigned int[NUMSTRM]; 
    float* penalty_sum = new float[NUMSTRM];
    thrust::fill(count, count+node_size, 0);
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
    thrust::fill(penalty_sum, penalty_sum+NUMSTRM, 0.0f);
    thrust::fill(max_index, max_index+NUMSTRM, 0);
    unsigned int* d_check;
    unsigned int* d_max;
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*(node_size+1)))){
        std::cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size)))){
        std::cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_track_val, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_check, sizeof(unsigned int)*NUMSTRM*node_size))){
        std::cout<<"Error allocating memory for d_check"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_penality, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_penality"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_max, sizeof(unsigned int)*NUMSTRM))){
        std::cout<<"Error allocating memory for d_max"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_values_temp, sizeof(float)*edge_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_values_temp"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*(node_size+1), cudaMemcpyHostToDevice))){
        std::cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_vec, vec, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying vec to device"<<endl;
    }
    delete[] vec;
    if(!HandleCUDAError(cudaMemcpy(d_res, res, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying res to device"<<endl;
    }
    delete[] res;

    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size*NUMSTRM)))){
        std::cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_values, values, sizeof(float)*edge_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying values to device"<<endl;
    }
    // delete[] values;


    float* rand_init;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_init, NUMSTRM*num_walker*sizeof(float)))){
        std::cout<<"Error allocating memory for rand_frog"<<endl;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /*Now, we have the random numbers generated*/
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    if(!HandleCUDAError(cudaMalloc((void**)&rand_vec_init, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for rand_vec_init"<<endl;
    } 
    if(!HandleCUDAError(cudaMemcpy(rand_vec_init, h_rand_vec_init, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying h_rand_vec_init to device"<<endl;
    }

    float* store_stream_res;
    float* h_store_stream_res = new float[node_size*NUMSTRM];
    if(!HandleCUDAError(cudaMalloc((void**)&store_stream_res, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for store_stream_res"<<endl;
    }
    thrust::fill(thrust::device, store_stream_res, store_stream_res+node_size*NUMSTRM, 0.0f);
    delete[] h_store_stream_res;
    float* rand_numbers;
    if (!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * edge_size*NUMSTRM))) {
        std::cout << "Error allocating memory for rand_numbers" << endl;
    }
    thrust::fill(thrust::device, d_penality, d_penality+node_size, 0.0f);
    printCudaMemoryUsage();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float tol_thresh=1;
    epochs = 500*(K/NUMSTRM+1);
    for(int i = 0; i<NUMSTRM;i++){
        thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+i*edge_size+edge_size, 1.0f/(1.0f*edge_size));
        thrust::fill(thrust::device.on(streams[i]), d_values_temp+i*edge_size, d_values_temp+i*edge_size+edge_size, 1.0f/(1.0f*edge_size));
    }
    for(int k = 0; k < epochs; k++){
        // std::cout<<"Epoch "<<k<<endl;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        thrust::fill(sum, sum+NUMSTRM, 0.0f);
        thrust::fill(temp_sum, temp_sum+NUMSTRM, 0.0f);
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_init, num_walker*NUMSTRM);
        for(int i = 0; i < NUMSTRM; i++){
            //Initialize the random vector
            float* rand_init_i = rand_init + i*num_walker;
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            unsigned int* d_check_i = d_check + i*node_size;
            thrust::fill(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, 0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_check_i, d_check_i+node_size, 0);
            thrust::fill(thrust::device.on(streams[i]),d_res+i*node_size, d_res+i*node_size+node_size, 0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_track_val+i*node_size, d_track_val+i*node_size+node_size, 0.0f);
            Init_Random<<<blocks_per_stream, TPB,0,streams[i]>>>(rand_vec_init_i, rand_init_i, node_size, num_walker);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device at Init Random for Stream "<<i<<endl;
            }
            thrust::copy(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, d_check_i);
            float* rand_numbers_i = rand_numbers + i*edge_size;
            float* d_values_i = d_values + i*edge_size;
            // curandGenerator_t gen;
            // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            // srand(time(0));
            // int rand_seed = rand();
            // curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            // curandGenerateUniform(gen, rand_numbers_i, edge_size);
            // curandDestroyGenerator(gen);
            thrust::copy(thrust::device.on(streams[i]), d_values_temp+i*edge_size, d_values_temp+i*edge_size+edge_size, d_values_i);
            // thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values_v2(x,y,threshold); });
        }
        int while_count = 0;
        bool check = true;
        while(check){
            while_count++;
            // cout<<"While count: "<<while_count<<endl;
            // float level_thresh = exp(-(powf(log(1-threshold),2.0f))/((2/3)*powf(log(1-threshold),2.0f)+(2/3)*log(1-threshold)+1));
            for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > 0){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    float* d_track_i = d_track_val + i*node_size;
                    unsigned int* d_check_i = d_check + i*node_size;
                    float* d_penality_i = d_penality + i*node_size;
                    sparseCSRMat_Vec_Mult_Mart_BFS<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i,d_check_i, threshold, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                        exit(0);
                    }
                    // thrust::transform(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, d_res_i, [=] __device__ (float x) { return x*level_thresh; });
                    Copy<float><<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i, rand_vec_init_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at Copy for stream "<<i<<endl;
                        exit(0);
                    }
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_track_i, rand_vec_init_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at Float_VectAdd for stream "<<i<<endl;
                        exit(0);
                    }
                    thrust::device_ptr<unsigned int> d_check_ptr(d_check_i);
                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), d_check_ptr, d_check_ptr+node_size);
                    tol[i] = sum[i]-temp_sum[i];
                    temp_sum[i] = sum[i];
                    thrust::fill(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, 0.0f);
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > 0){
                    check = true;
                }
            }
        }
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = d_track_val + i*node_size;
            float* r_v_i = rand_vec_init + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            float* d_penality_i = d_penality + i*node_size;
            //Take the softmax of rand_vec_init_i
            thrust::copy(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, d_penality_i);
            sum[i] = thrust::reduce(thrust::device.on(streams[i]), d_penality_i, d_penality_i+node_size);
            float temp = sum[i];
            // cout<<"Sum: "<<sum[i]<<endl;
            if(sum[i]>0){
                Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                }
                //Find the index of the maximum element in the vector
                float* iter = thrust::max_element(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                max_index[i] = iter - rand_vec_init_i;
                count[max_index[i]]++;
            }
            unsigned int nmstrm = NUMSTRM;
            // thrust::transform(thrust::device.on(streams[i]), d_penality, d_penality+node_size, d_penality, [=] __device__ (float x) { return (x/nmstrm); });
            thrust::fill(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, 0);
        }
        if(!HandleCUDAError(cudaMemcpy(d_max, max_index, sizeof(unsigned int)*NUMSTRM, cudaMemcpyHostToDevice))){
            std::cout<<"Error copying max_index to device"<<endl;
        }
        for(int i=0; i<NUMSTRM;i++){
            float* d_values_temp_i = d_values_temp + i*edge_size;
            Zero_Rows_Max_Idx<<<blocks_per_stream, TPB,0,streams[i]>>>(d_values_temp_i,d_csc,d_succ,d_max,node_size,NUMSTRM);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Zero_Rows_Max_Idx at stream "<<i<<endl;
            }
            Zero_Cols_Max_Idx<<<blocks_per_stream, TPB,0,streams[i]>>>(d_values_temp_i,d_csc,d_succ,d_max,node_size,edge_size,NUMSTRM);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Zero_Cols_Max_Idx at stream "<<i<<endl;
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    curandDestroyGenerator(gen);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file,blocks_per_stream, milliseconds, damping_factor, threshold, epochs);
    if(!HandleCUDAError(cudaFree(d_csc))){
        std::cout<<"Error freeing d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_succ))){
        std::cout<<"Error freeing d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_vec))){
        std::cout<<"Error freeing d_vec"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_res))){
        std::cout<<"Error freeing d_res"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_values))){
        std::cout<<"Error freeing d_values"<<endl;
    }
    unsigned int* h_rand_idx = new unsigned int[node_size];
    unsigned int* h_rand_score_idx = new unsigned int[node_size];
    thrust::fill(h_rand_idx, h_rand_idx+node_size, 0);
    // Take the sum of the vectors and then sort them
    for(int i = 1; i<NUMSTRM;i++){
        float* store_stream_res_i = store_stream_res + i*node_size;
        Float_VectAdd<<<blocks_per_stream, TPB>>>(store_stream_res, store_stream_res_i, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
        }
    }
    //Get the top k indexes
    float* h_store_stream_res_fin = new float[node_size];
    if(!HandleCUDAError(cudaMemcpy(h_store_stream_res_fin, store_stream_res, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying store_stream_res to host"<<endl;
    }
    thrust::sequence(h_rand_score_idx, h_rand_score_idx+node_size,0);
    thrust::sort_by_key(h_store_stream_res_fin, h_store_stream_res_fin+node_size, h_rand_score_idx, thrust::greater<float>());
    thrust::sequence(h_rand_idx, h_rand_idx+node_size,0);
    thrust::sort_by_key(count, count+node_size, h_rand_idx, thrust::greater<unsigned int>());
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_score_idx[i];
        cout<<count[i]<<endl;
    }
    Export_Counts(COUNT_PATH, count,h_rand_idx, node_size);
    Export_Scores(SCORE_PATH, h_store_stream_res_fin, h_rand_score_idx, node_size);
    // delete[] h_rand_idx;

    for(int i = 0; i<NUMSTRM;i++){
        if(!HandleCUDAError(cudaStreamDestroy(streams[i]))){
            std::cout<<"Error destroying stream number "<<i<<endl;
        }
    }
    if(!HandleCUDAError(cudaFree(store_stream_res))){
        std::cout<<"Error freeing store_stream_res"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_numbers))){
        std::cout<<"Error freeing rand_numbers"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_vec_init))){
        std::cout<<"Error freeing rand_vec_init"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_init))){
        std::cout<<"Error freeing rand_init"<<endl;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        std::cout<<"Error resetting device"<<endl;
    }
    delete[] h_rand_vec_init;
    delete[] l2_norm_rand_vec_init;
    delete[] l2_norm_d_res;
    delete[] sum;
    delete[] tol;
    delete[] values;

}

__host__ void  RIM_rand_Mart_BFS_v3(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, float threshold, string file){
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM;
    blocks_per_stream = 40;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int num_walker = node_size/(NUMSTRM)+1;
    unsigned int epochs;
    unsigned int* d_csc;
    unsigned int* d_succ;
    unsigned int* count = new unsigned int[node_size];
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* d_track_val;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* temp_sum = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    float* d_penality;
    float* d_values_temp;
    float* d_diff;
    float* d_res_temp;
    unsigned int* max_index = new unsigned int[NUMSTRM]; 
    float* penalty_sum = new float[NUMSTRM];
    unsigned int* d_idx;
    thrust::fill(count, count+node_size, 0);
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
    thrust::fill(penalty_sum, penalty_sum+NUMSTRM, 0.0f);
    thrust::fill(max_index, max_index+NUMSTRM, 0);
    unsigned int* d_check;
    unsigned int* d_max;
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*(node_size+1)))){
        std::cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size)))){
        std::cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_track_val, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_check, sizeof(unsigned int)*NUMSTRM*node_size))){
        std::cout<<"Error allocating memory for d_check"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_penality, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_penality"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_max, sizeof(unsigned int)*NUMSTRM))){
        std::cout<<"Error allocating memory for d_max"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_values_temp, sizeof(float)*edge_size*NUMSTRM))){
        std::cout<<"Error allocating memory for d_values_temp"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_diff, sizeof(float)*NUMSTRM*node_size))){
        std::cout<<"Error allocating memory for d_diff"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res_temp, sizeof(float)*NUMSTRM*node_size))){
        std::cout<<"Error allocating memory for d_res_temp"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_idx, sizeof(unsigned int)*NUMSTRM*node_size))){
        std::cout<<"Error allocating memory for d_idx"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*(node_size+1), cudaMemcpyHostToDevice))){
        std::cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_vec, vec, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying vec to device"<<endl;
    }
    delete[] vec;
    if(!HandleCUDAError(cudaMemcpy(d_res, res, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying res to device"<<endl;
    }
    delete[] res;

    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size*NUMSTRM)))){
        std::cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_values, values, sizeof(float)*edge_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying values to device"<<endl;
    }
    // delete[] values;


    float* rand_init;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_init, NUMSTRM*num_walker*sizeof(float)))){
        std::cout<<"Error allocating memory for rand_frog"<<endl;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    /*Now, we have the random numbers generated*/
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    if(!HandleCUDAError(cudaMalloc((void**)&rand_vec_init, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for rand_vec_init"<<endl;
    } 
    if(!HandleCUDAError(cudaMemcpy(rand_vec_init, h_rand_vec_init, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying h_rand_vec_init to device"<<endl;
    }

    float* store_stream_res;
    float* h_store_stream_res = new float[node_size*NUMSTRM];
    if(!HandleCUDAError(cudaMalloc((void**)&store_stream_res, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for store_stream_res"<<endl;
    }
    thrust::fill(thrust::device, store_stream_res, store_stream_res+node_size*NUMSTRM, 0.0f);
    delete[] h_store_stream_res;
    float* rand_numbers;
    if (!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * edge_size*NUMSTRM))) {
        std::cout << "Error allocating memory for rand_numbers" << endl;
    }
    thrust::fill(thrust::device, d_penality, d_penality+node_size, 1.0f);
    printCudaMemoryUsage();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float tol_thresh=1;
    // epochs = (K)*(node_size/(NUMSTRM)+1);
    // epochs = 500*(K/NUMSTRM+1);
    epochs = (node_size/(600*NUMSTRM)+1)*(K/2);
    cout<<"Epochs: "<<epochs<<endl;
    for(int i = 0; i<NUMSTRM;i++){
        //Fill the values list prior to the start of the algorithm
        thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+i*edge_size+edge_size, 1.0f);
        // thrust::fill(thrust::device.on(streams[i]), d_values_temp+i*edge_size, d_values_temp+i*edge_size+edge_size, 1.0f);
    }
    for(int k = 0; k < epochs; k++){
        // std::cout<<"Epoch "<<k<<endl;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        thrust::fill(sum, sum+NUMSTRM, 0.0f);
        thrust::fill(temp_sum, temp_sum+NUMSTRM, 0.0f);
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_init, num_walker*NUMSTRM);
        for(int i = 0; i < NUMSTRM; i++){
            //Initialize the random vector
            float* rand_init_i = rand_init + i*num_walker;
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            unsigned int* d_check_i = d_check + i*node_size;
            float* d_diff_i = d_diff + i*node_size;
            float* d_res_temp_i = d_res_temp + i*node_size;
            thrust::fill(thrust::device.on(streams[i]),d_res+i*node_size, d_res+i*node_size+node_size, 0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_track_val+i*node_size, d_track_val+i*node_size+node_size, 0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_diff_i,d_diff_i+node_size,0.0f);
            thrust::fill(thrust::device.on(streams[i]),d_res_temp_i,d_res_temp_i+node_size,0.0f);
            Init_Random<<<blocks_per_stream, TPB,0,streams[i]>>>(rand_vec_init_i, rand_init_i, node_size, num_walker);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device at Init Random for Stream "<<i<<endl;
            }
            thrust::copy(thrust::device.on(streams[i]),rand_vec_init_i,rand_vec_init_i+node_size, d_check_i);
        }
        int while_count = 0;
        bool check = true;
        while(check){
            while_count++;
            for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > 0){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    float* d_track_i = d_track_val + i*node_size;
                    unsigned int* d_check_i = d_check + i*node_size;
                    float* d_penality_i = d_penality + i*node_size;
                    float* d_diff_i = d_diff + i*node_size; 
                    float* d_res_temp_i = d_res_temp + i*node_size;
                    sparseCSRMat_Vec_Mult_Mart_BFS<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i,d_check_i, threshold, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                        exit(0);
                    }
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, d_res_temp+i*node_size);
                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), d_res_temp_i, d_res_temp_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, d_res_i, [=] __device__ (float x) { return x/temp; });
                    
                    thrust::transform(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i, d_diff_i, thrust::minus<float>());
                    thrust::transform(thrust::device.on(streams[i]), d_diff_i, d_diff_i+node_size, d_diff_i, [] __device__ (float x) { return x * x; });
                    tol[i] = sum[i]-temp_sum[i];
                    temp_sum[i] = sum[i];
                    // cout<<"Tol: "<<tol[i]<<endl;    
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);
                    thrust::fill(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, 0.0f);
                    thrust::fill(thrust::device.on(streams[i]), d_res_temp_i, d_res_temp_i+node_size, 0.0f);
                }
            }
            //In circumstance where we want all the be less than
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > 1e-5){
                    check = true;
                }
            }
            //In circumstance we want one to be less than
            // check = true;
            // for(int i=0;i<NUMSTRM;i++){
            //     if(tol[i] <= 1e-5){
            //         check = false;
            //     }
            // }
        }
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            float* d_penality_i = d_penality + i*node_size;
            float* d_res_temp_i = d_res_temp + i*node_size;
            unsigned int* d_idx_i = d_idx + i*node_size;
            //Take the softmax of rand_vec_init_i
            thrust::copy(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, d_res_temp_i);
            sum[i] = thrust::reduce(thrust::device.on(streams[i]), d_res_temp_i, d_res_temp_i+node_size);
            float temp = sum[i];
            // cout<<"Sum: "<<sum[i]<<endl;
            if(sum[i]>0){
                thrust::sequence(thrust::device.on(streams[i]), d_idx_i, d_idx_i+node_size,0);
                thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                //Find the index of the maximum element in the vector
                float* iter = thrust::max_element(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                max_index[i] = iter - rand_vec_init_i;
                count[max_index[i]]++;
                thrust::copy(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, d_penality_i);
                thrust::sort_by_key(thrust::device.on(streams[i]), d_penality_i, d_penality_i+node_size, d_idx_i, thrust::greater<float>());
                //Changes here
                Float_VectAdd_Cap<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i,d_idx_i, node_size, K);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                }
                thrust::fill(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, 0);

            }
            unsigned int nmstrm = NUMSTRM;
            // thrust::transform(thrust::device.on(streams[i]), d_penality, d_penality+node_size, d_penality, [=] __device__ (float x) { return (x/nmstrm); });
        }
        if(!HandleCUDAError(cudaMemcpy(d_max, max_index, sizeof(unsigned int)*NUMSTRM, cudaMemcpyHostToDevice))){
            std::cout<<"Error copying max_index to device"<<endl;
        }
        for(int i=0; i<NUMSTRM;i++){
            float* d_values_temp_i = d_values + i*edge_size;
            Zero_Rows_Max_Idx<<<blocks_per_stream, TPB,0,streams[i]>>>(d_values_temp_i,d_csc,d_succ,d_max,node_size,NUMSTRM);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Zero_Rows_Max_Idx at stream "<<i<<endl;
            }
            Zero_Cols_Max_Idx<<<blocks_per_stream, TPB,0,streams[i]>>>(d_values_temp_i,d_csc,d_succ,d_max,node_size,edge_size,NUMSTRM);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Zero_Cols_Max_Idx at stream "<<i<<endl;
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    curandDestroyGenerator(gen);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file,blocks_per_stream, milliseconds, damping_factor, threshold, epochs);
    if(!HandleCUDAError(cudaFree(d_csc))){
        std::cout<<"Error freeing d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_succ))){
        std::cout<<"Error freeing d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_vec))){
        std::cout<<"Error freeing d_vec"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_res))){
        std::cout<<"Error freeing d_res"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_values))){
        std::cout<<"Error freeing d_values"<<endl;
    }
    unsigned int* h_rand_idx = new unsigned int[node_size];
    unsigned int* h_rand_score_idx = new unsigned int[node_size];
    thrust::fill(h_rand_idx, h_rand_idx+node_size, 0);
    // Take the sum of the vectors and then sort them
    float* d_store_res_fin;
    if(!HandleCUDAError(cudaMalloc((void**)&d_store_res_fin, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_store_res_fin"<<endl;
    }
    thrust::fill(thrust::device, d_store_res_fin, d_store_res_fin+node_size, 0.0f);
    Condense_Score<<<blocks_per_stream, TPB>>>(d_store_res_fin, store_stream_res, node_size, NUMSTRM);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        std::cout<<"Error synchronizing device for Condense_Score"<<endl;
    }
    //Get the top k indexes
    float* h_store_stream_res_fin = new float[node_size*NUMSTRM];
    if(!HandleCUDAError(cudaMemcpy(h_store_stream_res_fin, d_store_res_fin, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying store_stream_res to host"<<endl;
    }
    // thrust::sequence(h_rand_score_idx, h_rand_score_idx+node_size,0);
    // thrust::sort_by_key(h_store_stream_res_fin, h_store_stream_res_fin+node_size, h_rand_score_idx, thrust::greater<float>());
    thrust::sequence(h_rand_idx, h_rand_idx+node_size,0);
    ValueTuple* count_tuple = new ValueTuple[node_size];
    Make_Tuple_Count(count,h_rand_idx,count_tuple, node_size);
    thrust::sort_by_key(h_store_stream_res_fin,h_store_stream_res_fin+node_size,count_tuple, thrust::greater<float>());
    // thrust::default_random_engine g;
    // thrust::shuffle(thrust::host, count, count + node_size, g);
    //Need to find a way to shuffle the indexes that have the same values, shuffle based on score first, then by count
    Split_Tuple_Count(count, h_rand_idx, count_tuple, node_size);
    Export_Scores(SCORE_PATH, h_store_stream_res_fin, h_rand_idx, node_size);
    thrust::sort_by_key(count, count+node_size, h_rand_idx, thrust::greater<unsigned int>());

    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
        cout<<count[i]<<endl;
    }
    Export_Counts(COUNT_PATH, count,h_rand_idx, node_size);
    // delete[] h_rand_idx;

    for(int i = 0; i<NUMSTRM;i++){
        if(!HandleCUDAError(cudaStreamDestroy(streams[i]))){
            std::cout<<"Error destroying stream number "<<i<<endl;
        }
    }
    if(!HandleCUDAError(cudaFree(store_stream_res))){
        std::cout<<"Error freeing store_stream_res"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_numbers))){
        std::cout<<"Error freeing rand_numbers"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_vec_init))){
        std::cout<<"Error freeing rand_vec_init"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_init))){
        std::cout<<"Error freeing rand_init"<<endl;
    }
    if(!HandleCUDAError(cudaDeviceReset())){
        std::cout<<"Error resetting device"<<endl;
    }
    delete[] h_rand_vec_init;
    delete[] l2_norm_rand_vec_init;
    delete[] l2_norm_d_res;
    delete[] sum;
    delete[] tol;
    delete[] values;

}

template <typename T>
__global__ void Copy(T* d_res, T* d_vec, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < node_size; i += blockDim.x * gridDim.x){
        d_vec[i] = d_res[i];
    }
}

__global__ void Transform_Bool(float* d_res, unsigned int* d_vec, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < node_size; i += blockDim.x * gridDim.x){
        if(abs(d_res[i]) > 0.0f){
            d_vec[i] = 1;
            // printf("d_res[i]: %f\n", d_res[i]);
        }
        else{
            d_vec[i] = 0;
            // printf("d_res[i]: %f\n", d_res[i]);
        }
    }
}

__global__ void Calc_Penalty(float* d_res, float* d_penality, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < node_size; i += blockDim.x * gridDim.x){
        d_penality[i] += 1-d_res[i];
    }
}

__global__ void Condense_Score(float* fin, float* inter, unsigned int node_size, unsigned int num_strm){
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = idx; i < node_size; i += blockDim.x * gridDim.x){
        for(int j = 0; j < num_strm; j++){
            fin[i] += inter[i+j*node_size];
        }
    }
}