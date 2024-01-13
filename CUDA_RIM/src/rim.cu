#include "../include/data.h"


__device__ float eval_values(float rand_num, float val,float threshold){
    if(rand_num > threshold){
        return val;
    }
    else{
        return 0.0f;
    }
}

__device__ float eval_values_v2(float rand_num, float val,float threshold){
    if(rand_num > threshold){
        return rand_num;
    }
    else{
        return 0.0f;
    }
}



__host__ void Save_Data(string file, float time, float damping_factor, float threshold,unsigned int epoch){
    ofstream myfile;
    myfile.open(file, std::ios_base::app);
    myfile<<NUMSTRM<<","<<time<<","<<damping_factor<<","<<threshold<<","<<epoch<<","<<K;
    myfile.close();
}

__host__ void  RIM_rand_Ver1(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file){
    float threshold = .2;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int num_walker = node_size/20;
    unsigned int epochs=30;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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


    // // unsigned int num_blocks = (node_size+TPB-1)/TPB;
    // // unsigned int num_blocks2 = (edge_size+TPB-1)/TPB;
    float* rand_init;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_init, NUMSTRM*num_walker*sizeof(float)))){
        std::cout<<"Error allocating memory for rand_frog"<<endl;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    srand(time(0));
    int rand_seed = rand();
    curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
    curandGenerateUniform(gen, rand_init, num_walker*NUMSTRM);
    /*Now, we have the random numbers generated*/
    curandDestroyGenerator(gen);
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    if(!HandleCUDAError(cudaMalloc((void**)&rand_vec_init, sizeof(float)*node_size*NUMSTRM))){
        std::cout<<"Error allocating memory for rand_vec_init"<<endl;
    } 
    if(!HandleCUDAError(cudaMemcpy(rand_vec_init, h_rand_vec_init, sizeof(float)*node_size*NUMSTRM, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying h_rand_vec_init to device"<<endl;
    }
    float* rand_numbers;
    if (!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * edge_size*NUMSTRM))) {
        std::cout << "Error allocating memory for rand_numbers" << endl;
    }
    printCudaMemoryUsage();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < NUMSTRM; i++){
        //Initialize the random vector
        float* rand_init_i = rand_init + i*num_walker;
        float* rand_vec_init_i = rand_vec_init + i*node_size;
        Init_Random<<<blocks_per_stream, TPB,0,streams[i]>>>(rand_vec_init_i, rand_init_i, node_size, num_walker);
        if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
            std::cout<<"Error synchronizing device at Init Random for Stream "<<i<<endl;
        }
    }
    for(int i = 0; i < epochs; i++){
        std::cout<<"Epoch "<<i<<endl;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        int while_count = 0;
        while_count=0;
        while(thrust::all_of(thrust::host, tol, tol+NUMSTRM, [=] __device__ (float x) { return x > threshold; }) && while_count < MAX_WHILE){
            while_count++;
            for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_numbers_i = rand_numbers + i*NUMSTRM;
                    curandGenerator_t gen;
                    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                    srand(time(0));
                    int rand_seed = rand();
                    curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
                    curandGenerateUniform(gen, rand_numbers_i, edge_size);
                    curandDestroyGenerator(gen);
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                    }
                    thrust::fill(thrust::device.on(streams[i]), d_values_i, d_values_i+edge_size, 1.0f);
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > threshold){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                    }
                    //Need to normalize the vector using thrust library

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                    // thrust::fill(thrust::device.on(streams[i]), d_vec, d_vec+node_size, 1.0f/node_size);
                }
            }
        }
        // switch(epochs%NUMSTRM){
        //     case 0:
        //         thrust::transform(thrust::device, rand_vec_init+(NUMSTRM-1)*edge_size, rand_vec_init+(NUMSTRM)*edge_size, rand_vec_init, [damping_factor] __device__ (float x) { return x + damping_factor * x; });
        //         break;
        //     default:
        //         thrust::transform(thrust::device, rand_vec_init+(epochs%NUMSTRM)*edge_size, rand_vec_init+(1+epochs%NUMSTRM)*edge_size+edge_size, rand_vec_init+(1+epochs%NUMSTRM)*edge_size, [damping_factor] __device__ (float x) { return x + damping_factor * x; });
        //         break;
            
        // }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
        thrust::transform(thrust::device, rand_vec_init, rand_vec_init+node_size, rand_vec_init+i*node_size, rand_vec_init, thrust::plus<float>());
    }
    thrust::sort_by_key(thrust::device, rand_vec_init, rand_vec_init+node_size, rand_idx, thrust::greater<float>());
    //Get the top k indexes
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
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
}


__host__ void  RIM_rand_Ver2(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file){
    float threshold = .3;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int num_walker = node_size/20;
    unsigned int epochs=30;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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
    float tol_thresh=1e-6;
    for(int i = 0; i < epochs; i++){
        std::cout<<"Epoch "<<i<<endl;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
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
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
            for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                    }
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > threshold){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* store_stream_res_i = store_stream_res + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                    }
                    //Need to normalize the vector using thrust library

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                    }
                    // thrust::fill(thrust::device.on(streams[i]), d_vec, d_vec+node_size, 1.0f/node_size);
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > tol_thresh){
                    check = true;
                }
            }
        }
        for(int i = 0; i<NUMSTRM;i++){
            thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+(i+1)*edge_size, 1.0f);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    curandDestroyGenerator(gen);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
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


__host__ void  RIM_rand_Ver3_PR(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, edge* edge_list, string file, string pr_file){
    float threshold = .4;
    float tol_thresh = 1e-4;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    float* pr_vector = new float[node_size];
    float* pr_time = new float[1];
    *pr_time = 0.0f;
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    thrust::fill(pr_vector, pr_vector+node_size, 0.0f);
    int *pr_csc, *pr_succ;
    pr_csc = new int[node_size+1];
    pr_succ = new int[edge_size];
    thrust::copy(csc,csc+node_size+1,pr_csc);
    thrust::copy(succ,succ+edge_size,pr_succ);
    // PageRank(pr_vector,csc,succ,.15f,node_size,edge_size,100,1e-6,pr_time);
    PageRank_Sparse(pr_vector,.15f,node_size,edge_size,100,1e-6,pr_time,pr_file);
    float* d_pr;
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_pr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_pr, pr_vector, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying pr_vector to device"<<endl;
    }
    delete[] pr_vector;
    unsigned int epochs=30;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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
    for(int i = 0; i < epochs; i++){
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        for(int i =0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            thrust::copy(thrust::device.on(streams[i]), d_pr, d_pr+node_size, rand_vec_init_i);
            //Initialize the random vector
            float* rand_numbers_i = rand_numbers + i*edge_size;
            float* d_values_i = d_values + i*edge_size;
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            srand(time(0));
            int rand_seed = rand();
            curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            curandGenerateUniform(gen, rand_numbers_i, edge_size);
            curandDestroyGenerator(gen);
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        int while_count = 0;
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
           for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                        exit(1);
                    }
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > tol_thresh){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* store_stream_res_i = store_stream_res + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                        exit(1);
                    }
                    //Need to normalize the vector using thrust library

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > tol_thresh){
                    check = true;
                }
            }
        }
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                exit(1);
            }
            thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+(i+1)*edge_size, 1.0f);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds += *pr_time;
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
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

__host__ void  RIM_rand_Ver4_Greedy(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file){
    float threshold = .3;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int num_walker = node_size/20;
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    float* pr_vector = new float[node_size];
    float* pr_time = new float[1];
    *pr_time = 0.0f;
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    thrust::fill(pr_vector, pr_vector+node_size, 0.0f);
    // PageRank(pr_vector,csc,succ,.15f,node_size,edge_size,100,1e-6,pr_time);
    PageRank_Sparse(pr_vector,.15f,node_size,edge_size,100,1e-6,pr_time,pr_file);
    float* d_pr;
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_pr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_pr, pr_vector, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying pr_vector to device"<<endl;
    }
    delete[] pr_vector;
    unsigned int epochs=30;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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
    float* rand_init;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_init, NUMSTRM*num_walker*sizeof(float)))){
        std::cout<<"Error allocating memory for rand_frog"<<endl;
    }
    if (!HandleCUDAError(cudaMalloc((void**)&rand_numbers, sizeof(float) * edge_size*NUMSTRM))) {
        std::cout << "Error allocating memory for rand_numbers" << endl;
    }
    unsigned int* d_seed_set;
    if(!HandleCUDAError(cudaMalloc((void**)&d_seed_set, sizeof(unsigned int)*K))){
        std::cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    thrust::fill(thrust::device, d_seed_set, d_seed_set+K, node_size+1);
    unsigned int fill_count = 0;
    unsigned int epoch = 0;
    printCudaMemoryUsage();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    curandGenerator_t gen;
    float tol_thresh=1e-4;
    while(fill_count < K){
        std::cout<<"Epoch "<<epoch<<endl;
        epoch++;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
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
            Init_Random<<<blocks_per_stream, TPB,0,streams[i]>>>(rand_vec_init_i, rand_init_i, node_size, num_walker);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device at Init Random for Stream "<<i<<endl;
            }

            float* rand_numbers_i = rand_numbers + i*NUMSTRM;
            float* d_values_i = d_values + i*edge_size;
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            srand(time(0));
            int rand_seed = rand();
            curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            curandGenerateUniform(gen, rand_numbers_i, edge_size);
            curandDestroyGenerator(gen);
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        while_count = 0;
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
            for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_numbers_i = rand_numbers + i*NUMSTRM;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                    }
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > threshold){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* store_stream_res_i = store_stream_res + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                    }
                    //Need to normalize the vector using thrust library

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > tol_thresh){
                    check = true;
                }
            }
        }
        cout<<"fill_count: "<<fill_count<<endl;
        for(int i = 0; i<NUMSTRM;i++){
            thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+(i+1)*edge_size, 1.0f);
            //Evaluate the maximum value in each rand_vec_init and save the index to d_seed_set
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            float* iter =thrust::max_element(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
            unsigned int index = iter - rand_vec_init_i;
            //check if index is already in d_seed_set
            unsigned int* iter2 = thrust::find(thrust::device.on(streams[i]), d_seed_set, d_seed_set+K, index);
            if(iter2 == d_seed_set+K){
                //index is not in d_seed_set
                thrust::fill(thrust::device.on(streams[i]), d_seed_set+fill_count, d_seed_set+fill_count+1, index);
                fill_count++;
                for(int j=0; j<NUMSTRM;j++){
                    float* d_values_i = d_values + j*edge_size;
                    Zero_Rows<<<blocks_per_stream, TPB,0,streams[j]>>>(d_values_i,d_csc,d_succ,d_seed_set,node_size,fill_count);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[j]))){
                        std::cout<<"Error synchronizing device for Zero_Rows at stream "<<i<<endl;
                    }
                }
            }
        }
    }
    curandDestroyGenerator(gen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds += *pr_time;
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
    for(int i = 0; i<NUMSTRM;i++){
        if(!HandleCUDAError(cudaStreamDestroy(streams[i]))){
            std::cout<<"Error destroying stream number "<<i<<endl;
        }
    }
    if(!HandleCUDAError(cudaMemcpy(seed_set, d_seed_set, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying d_seed_set to host"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_numbers))){
        std::cout<<"Error freeing rand_numbers"<<endl;
    }
    if(!HandleCUDAError(cudaFree(rand_vec_init))){
        std::cout<<"Error freeing rand_vec_init"<<endl;
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


__host__ void  RIM_rand_Ver5_Sig(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file){
    float threshold = .4;
    float tol_thresh = 1e-4;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    float* pr_vector = new float[node_size];
    float* pr_time = new float[1];
    *pr_time = 0.0f;
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    thrust::fill(pr_vector, pr_vector+node_size, 0.0f);
    int *pr_csc, *pr_succ;
    pr_csc = new int[node_size+1];
    pr_succ = new int[edge_size];
    thrust::copy(csc,csc+node_size+1,pr_csc);
    thrust::copy(succ,succ+edge_size,pr_succ);
    // PageRank(pr_vector,csc,succ,.15f,node_size,edge_size,100,1e-6,pr_time);
    PageRank_Sparse(pr_vector,.15f,node_size,edge_size,100,1e-6,pr_time,pr_file);
    float* d_pr;
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_pr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_pr, pr_vector, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying pr_vector to device"<<endl;
    }
    delete[] pr_vector;
    unsigned int epochs=0;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    float* store_sum = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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
    float total_thresh = 1e-3;
    float temp_total_tol = 100.0f;
    float total_tol = 0.0f;
    float dif = 100.0f;
    while(dif > total_thresh && epochs<50){
        epochs++;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        for(int i =0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            thrust::copy(thrust::device.on(streams[i]), d_pr, d_pr+node_size, rand_vec_init_i);
            //Initialize the random vector
            float* rand_numbers_i = rand_numbers + i*edge_size;
            float* d_values_i = d_values + i*edge_size;
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            srand(time(0));
            int rand_seed = rand();
            curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            curandGenerateUniform(gen, rand_numbers_i, edge_size);
            curandDestroyGenerator(gen);
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        int while_count = 0;
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
           for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                        exit(1);
                    }
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > tol_thresh){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* store_stream_res_i = store_stream_res + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                        exit(1);
                    }
                    //Need to normalize the vector using thrust library
                    thrust::transform(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, d_res_i, [=] __device__ (float x) { return 1/(1+exp(-x)); });

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                    //store_sum[i] = thrust::reduce(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size);
                    // thrust::transform(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size, store_stream_res_i, [=] __device__ (float x) { return 1/(1+exp(-x)); });
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > tol_thresh){
                    check = true;
                }
            }
        }
        total_tol = 0.0f;
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                exit(1);
            }
            //Need to normalize the vector using thrust library
            store_sum[i] = thrust::reduce(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size);
            float temp = store_sum[i];
            thrust::transform(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size, store_stream_res_i, [=] __device__ (float x) { return x/temp; });
            thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+(i+1)*edge_size, 1.0f);
            total_tol += sqrtf(thrust::transform_reduce(thrust::device.on(streams[i]),store_stream_res+i*node_size,store_stream_res+(i+1)*node_size,[] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>()));
        }
        dif = abs(total_tol-temp_total_tol);
        temp_total_tol = total_tol;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds += *pr_time;
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
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


__host__ void  RIM_rand_Ver6_Tanh(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file){
    float threshold = .4;
    float tol_thresh = 1e-4;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    float* pr_vector = new float[node_size];
    float* pr_time = new float[1];
    *pr_time = 0.0f;
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    thrust::fill(pr_vector, pr_vector+node_size, 0.0f);
    int *pr_csc, *pr_succ;
    pr_csc = new int[node_size+1];
    pr_succ = new int[edge_size];
    thrust::copy(csc,csc+node_size+1,pr_csc);
    thrust::copy(succ,succ+edge_size,pr_succ);
    // PageRank(pr_vector,csc,succ,.15f,node_size,edge_size,100,1e-6,pr_time);
    PageRank_Sparse(pr_vector,.15f,node_size,edge_size,100,1e-6,pr_time,pr_file);
    float* d_pr;
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_pr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_pr, pr_vector, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying pr_vector to device"<<endl;
    }
    delete[] pr_vector;
    unsigned int epochs=0;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    float* store_sum = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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
    float total_thresh = 1e-3;
    float temp_total_tol = 100.0f;
    float total_tol = 0.0f;
    float dif = 100.0f;
    while(dif > total_thresh && epochs<50){
        epochs++;
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        for(int i =0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            thrust::copy(thrust::device.on(streams[i]), d_pr, d_pr+node_size, rand_vec_init_i);
            //Initialize the random vector
            float* rand_numbers_i = rand_numbers + i*edge_size;
            float* d_values_i = d_values + i*edge_size;
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            srand(time(0));
            int rand_seed = rand();
            curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            curandGenerateUniform(gen, rand_numbers_i, edge_size);
            curandDestroyGenerator(gen);
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        int while_count = 0;
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
           for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                        exit(1);
                    }
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > tol_thresh){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* store_stream_res_i = store_stream_res + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                        exit(1);
                    }
                    //Need to normalize the vector using thrust library
                    thrust::transform(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, d_res_i, [=] __device__ (float x) { return (exp(x)-exp(-x))/(exp(x)+exp(-x)); });

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                    //store_sum[i] = thrust::reduce(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size);
                    // thrust::transform(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size, store_stream_res_i, [=] __device__ (float x) { return 1/(1+exp(-x)); });
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > tol_thresh){
                    check = true;
                }
            }
        }
        total_tol = 0.0f;
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                exit(1);
            }
            //Need to normalize the vector using thrust library
            store_sum[i] = thrust::reduce(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size);
            float temp = store_sum[i];
            thrust::transform(thrust::device.on(streams[i]), store_stream_res_i, store_stream_res_i+node_size, store_stream_res_i, [=] __device__ (float x) { return x/temp; });
            thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+(i+1)*edge_size, 1.0f);
            total_tol += sqrtf(thrust::transform_reduce(thrust::device.on(streams[i]),store_stream_res+i*node_size,store_stream_res+(i+1)*node_size,[] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>()));
        }
        dif = abs(total_tol-temp_total_tol);
        temp_total_tol = total_tol;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds += *pr_time;
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
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


__host__ void  RIM_rand_Ver7_PR_Rand(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set, string file, string pr_file){
    float threshold = .4;
    float tol_thresh = 1e-4;
    float damping_factor =.3;
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Get the current device
    cudaGetDeviceProperties(&prop, device);  // Get the properties of the device

    int maxActiveBlocksPerMultiprocessor = prop.maxThreadsPerMultiProcessor / TPB;
    int maxActiveBlocks = prop.multiProcessorCount * maxActiveBlocksPerMultiprocessor;
    int blocks_per_stream = maxActiveBlocks/NUMSTRM+1;

    printf("Max active blocks: %d\n", maxActiveBlocks);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            std::cout<<"Error creating stream number "<<i<<endl;
        }
    }
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size*NUMSTRM];
    float* pr_vector = new float[node_size];
    float* pr_time = new float[1];
    *pr_time = 0.0f;
    thrust::fill(h_rand_vec_init, h_rand_vec_init+node_size*NUMSTRM, 0.0f);
    thrust::fill(pr_vector, pr_vector+node_size, 0.0f);
    int *pr_csc, *pr_succ;
    pr_csc = new int[node_size+1];
    pr_succ = new int[edge_size];
    thrust::copy(csc,csc+node_size+1,pr_csc);
    thrust::copy(succ,succ+edge_size,pr_succ);
    // PageRank(pr_vector,csc,succ,.15f,node_size,edge_size,100,1e-6,pr_time);
    PageRank_Sparse(pr_vector,.15f,node_size,edge_size,100,1e-6,pr_time,pr_file);
    float* d_pr;
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_pr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_pr, pr_vector, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying pr_vector to device"<<endl;
    }
    delete[] pr_vector;
    unsigned int epochs=30;
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[NUMSTRM*node_size];
    float* values = new float[NUMSTRM*edge_size];
    float* res = new float[NUMSTRM*node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_res = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+NUMSTRM*node_size, 0.0f);
    thrust::fill(vec, vec+NUMSTRM*node_size, 1.0f/node_size);
    thrust::fill(values, values+NUMSTRM*edge_size, 1.0f);
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
    for(int i = 0; i < epochs; i++){
        thrust::fill(tol,tol+NUMSTRM, 100.0f);
        for(int i =0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            thrust::copy(thrust::device.on(streams[i]), d_pr, d_pr+node_size, rand_vec_init_i);
            //Initialize the random vector
            float* rand_numbers_i = rand_numbers + i*edge_size;
            float* d_values_i = d_values + i*edge_size;
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            srand(time(0));
            int rand_seed = rand();
            curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
            curandGenerateUniform(gen, rand_numbers_i, edge_size);
            curandDestroyGenerator(gen);
            thrust::transform(thrust::device.on(streams[i]), rand_numbers_i, rand_numbers_i+edge_size, d_values_i, d_values_i, [threshold] __device__ (float x, float y) { return eval_values(x,y,threshold); });
        }
        int while_count = 0;
        bool check = true;
        while(check && while_count < MAX_WHILE){
            while_count++;
           for(int i = 0; i < NUMSTRM; i++){
                //Perform the first iteration of the algorithm
                if(tol[i] > threshold){
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* d_res_i = d_res + i*node_size;
                    float* d_values_i = d_values + i*edge_size;
                    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values_i, rand_vec_init_i, d_res_i, node_size);  
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                        exit(1);
                    }
                }
            }
            for(int i = 0; i < NUMSTRM; i++){
                // Add 1/n to the vector
                if(tol[i] > tol_thresh){
                    float* d_res_i = d_res + i*node_size;
                    float* d_vec_i = d_vec + i*node_size;
                    float* rand_vec_init_i = rand_vec_init + i*node_size;
                    float* store_stream_res_i = store_stream_res + i*node_size;
                    Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_res_i,d_vec_i, node_size);
                    if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                        std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                        exit(1);
                    }
                    //Need to normalize the vector using thrust library

                    l2_norm_d_res[i] = thrust::transform_reduce(thrust::device.on(streams[i]), d_res_i, d_res_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_d_res[i] = sqrt(l2_norm_d_res[i]);

                    l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                    l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);

                    tol[i] = abs(l2_norm_d_res[i]-l2_norm_rand_vec_init[i]);
                    thrust::copy(thrust::device.on(streams[i]), d_res_i, d_res_i+node_size, rand_vec_init_i);


                    sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size);
                    float temp = sum[i];
                    thrust::transform(thrust::device.on(streams[i]), rand_vec_init_i, rand_vec_init_i+node_size, rand_vec_init_i, [=] __device__ (float x) { return x/temp; });
                }
            }
            check = false;
            for(int i=0;i<NUMSTRM;i++){
                if(tol[i] > tol_thresh){
                    check = true;
                }
            }
        }
        for(int i = 0; i<NUMSTRM;i++){
            float* rand_vec_init_i = rand_vec_init + i*node_size;
            float* store_stream_res_i = store_stream_res + i*node_size;
            Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(store_stream_res_i,rand_vec_init_i, node_size);
            if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                std::cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                exit(1);
            }
            thrust::fill(thrust::device.on(streams[i]), d_values+i*edge_size, d_values+(i+1)*edge_size, 1.0f);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds += *pr_time;
    std::cout<<"Time taken: "<<milliseconds<<endl;
    Save_Data(file, milliseconds, damping_factor, threshold, epochs);
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
    if(!HandleCUDAError(cudaMemcpy(h_rand_idx, rand_idx, sizeof(unsigned int)*K, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying rand_idx to host"<<endl;
    }
    for(int i = 0; i < K; i++){
        seed_set[i] = h_rand_idx[i];
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


__host__ void CheckSparseMatVec(unsigned int* csc, unsigned int* succ,edge* edge_list, unsigned int node_size, unsigned int edge_size){
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[node_size];
    float* values = new float[edge_size];
    float* res = new float[node_size]; 
    thrust::fill(res, res+node_size, 0.0f);
    thrust::fill(vec, vec+node_size, 1.0f/node_size);
    thrust::fill(values, values+edge_size, 1.0f);
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
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*(node_size+1)))){
        std::cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size)))){
        std::cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_vec, vec, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying vec to device"<<endl;
    }
    delete[] vec;
    if(!HandleCUDAError(cudaMemcpy(d_res, res, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying res to device"<<endl;
    }
    delete[] res;
    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size)))){
        std::cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_values, values, sizeof(float)*edge_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying values to device"<<endl;
    }
    // delete[] values;
    // if(!HandleCUDAError(cudaMemcpy(values,d_values, sizeof(float)*edge_size, cudaMemcpyDeviceToHost))){
    //     std::cout<<"Error copying values to device"<<endl;
    // }

    float* rand_init;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_init, K*sizeof(float)))){
        std::cout<<"Error allocating memory for rand_frog"<<endl;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    srand(time(0));
    int rand_seed = rand();
    curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
    curandGenerateUniform(gen, rand_init, K);
    /*Now, we have the random numbers generated*/
    curandDestroyGenerator(gen);
    float* rand_vec_init;
    float* h_rand_vec_init = new float[node_size]{0.0f};
    if(!HandleCUDAError(cudaMalloc((void**)&rand_vec_init, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for rand_vec_init"<<endl;
    } 
    if(!HandleCUDAError(cudaMemcpy(rand_vec_init, h_rand_vec_init, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        std::cout<<"Error copying h_rand_vec_init to device"<<endl;
    }
    //Initialize the random vector
    Init_Random<<<blocks_per_stream, TPB>>>(rand_vec_init, rand_init, node_size, K);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        std::cout<<"Error synchronizing device"<<endl;
    }
    //Perform the first iteration of the algorithm
    sparseCSRMat_Vec_Mult<unsigned int><<<blocks_per_stream, TPB>>>(d_csc, d_succ, d_values, rand_vec_init, d_res, node_size);  
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        std::cout<<"Error synchronizing device"<<endl;
    }
    Float_VectAdd<<<blocks_per_stream, TPB>>>(d_res, d_vec, node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        std::cout<<"Error synchronizing device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_rand_vec_init, rand_vec_init, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying d_vec to host"<<endl;
    }
    float* h_res_GPU = new float[node_size]{0.0f};

    if(!HandleCUDAError(cudaMemcpy(h_res_GPU, d_res, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying d_vec to host"<<endl;
    }

    float* h_res_CPU = new float[node_size]{0.0f};
    float* A = (float*)malloc(sizeof(float)*node_size*node_size);
    GenAdj(edge_list, A, node_size, edge_size);
    h_MatVecMult(A, h_rand_vec_init, h_res_CPU, node_size);
    float* support_vec = new float[node_size];
    thrust::fill(support_vec, support_vec+node_size, 1.0f/node_size);
    for(int i = 0; i < node_size; i++){
        h_res_CPU[i] += support_vec[i];
    }
    float sum = 0.0f;
    sum = thrust::inner_product(thrust::device, d_res, d_res+node_size, d_res, 0.0f);
    sum = sqrt(sum);
    thrust::transform(thrust::device, d_res, d_res+node_size, d_res, thrust::placeholders::_1/sum);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        std::cout<<"Error synchronizing device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_res_GPU, d_res, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        std::cout<<"Error copying d_vec to host"<<endl;
    }
    Normalize_L2(h_res_CPU, node_size);
    Verify(h_res_GPU, h_res_CPU, node_size);

}


__global__ void Float_VectAdd(float* vec1, float* vec2, unsigned int size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < size){
        vec1[tid] += vec2[tid];
    }
}

__global__ void Init_Random(float* vec, float* rand_init, unsigned int size, unsigned int k){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int idx =0;
    if(tid<k){
        rand_init[tid] = floorf(rand_init[tid]*size);
        idx = (unsigned int)rand_init[tid];
        idx = idx%size;
        vec[idx] = 1.0f;
    }
}

__host__ void Verify(float* gpu_vec, float* cpu_vec, unsigned int size){
    float sum = 0.0f;
    for(int i = 0; i < size; i++){
        sum = abs(gpu_vec[i]-cpu_vec[i]);
        if (sum > 0.00001f){
            std::cout<<"Error at index "<<i<<endl;
            std::cout<<"GPU: "<<gpu_vec[i]<<endl;
            std::cout<<"CPU: "<<cpu_vec[i]<<endl;
            return;
        }
    }
    std::cout<<"No errors found"<<endl;
}

__global__ void Zero_Rows(float* values, unsigned int* csc, unsigned int* succ, unsigned int* idx, unsigned int node_size, unsigned int num_cancel){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid<num_cancel){
        unsigned int int_idx =idx[tid];
        unsigned int start = csc[int_idx];
        unsigned int end = csc[int_idx+1];
        for(int i = start; i < end; i++){
            values[i] = 0.0f;
        }
    }
}

__host__ void Verify_Pr(float* sparse_vec, float* full_vec, unsigned int node_size){
    for(int i = 0; i<node_size;i++){
        if(abs(sparse_vec[i]- full_vec[i])>=1e-6){
            std::cout<<"Error at index "<<i<<endl;
            std::cout<<"Sparse: "<<sparse_vec[i]<<endl;
            std::cout<<"Full: "<<full_vec[i]<<endl;
            return;
        }
    }
}
