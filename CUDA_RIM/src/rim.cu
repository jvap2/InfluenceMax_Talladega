#include "../include/data.h"
/*The code for this come from Tang et al IMM Algorithm
The psuedo code is as follows:
1) Initialize a set R={empty set} and an integer LB = 1
2) Let epsilon'=sqrt(2)epsilon
3) for i = 1 to log_2(n)-1 do:
4) 	Let x=n/2^i
5) 	Let theta = lambda'/x
6)  while |R|<theta_i do:
7)      Select a node from G uniformly at random
8)      Generate an RR set for v, and insert it into R
9)  Let S_i = NodeSelection(R)
10) if n*FR(S)>=(1+epsilon')*x then:
11)     LB = n*FR(S)/(1+epsilon')
12)     break
13) Let theta= lambda^{*}/LB
14) while |R|<theta do:
15)     Select a node from G uniformly at random
16)     Generate an RR set for v, and insert it into R
17) Return R


What is going to be needed:
1) A graph
2) A set of RR sets
3) A set of nodes
4) A set of edges

How do we want to store these:
1) The RR sets would be traversable with CSC format, and any forward traversal will need
but would it be convenient to traverse both ways with a COO format?
2) We may be able to use linked lists as well but this will be slower, but also harder to implement, and harder for generating RRR sets

*/


__host__ void  RIM_rand_Ver1(unsigned int* csc, unsigned int* succ, unsigned int node_size, unsigned int edge_size, unsigned int* seed_set){
    float threshold = 0.75;
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
            cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    float* d_res;
    float* vec = new float[node_size];
    float* values = new float[edge_size];
    float* res = new float[node_size]; 
    float* tol = new float[NUMSTRM];
    float* sum = new float[NUMSTRM];
    float* l2_norm_d_vec = new float[NUMSTRM];
    float* l2_norm_rand_vec_init = new float[NUMSTRM];
    thrust::fill(sum, sum+NUMSTRM, 0.0f);
    thrust::fill(tol,tol+NUMSTRM, 100.0f);
    thrust::fill(res, res+node_size, 0.0f);
    thrust::fill(vec, vec+node_size, 1.0f/node_size);
    thrust::fill(values, values+edge_size, 1.0f);
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*(node_size+1)))){
        cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size)))){
        cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size))){
        cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res, sizeof(float)*node_size))){
        cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_vec, vec, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying vec to device"<<endl;
    }
    delete[] vec;
    if(!HandleCUDAError(cudaMemcpy(d_res, res, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying res to device"<<endl;
    }
    delete[] res;
    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size)))){
        cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_values, values, sizeof(float)*edge_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying values to device"<<endl;
    }
    // delete[] values;
    if(!HandleCUDAError(cudaMemcpy(values,d_values, sizeof(float)*edge_size, cudaMemcpyDeviceToHost))){
        cout<<"Error copying values to device"<<endl;
    }

    // unsigned int num_blocks = (node_size+TPB-1)/TPB;
    // unsigned int num_blocks2 = (edge_size+TPB-1)/TPB;
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
        cout<<"Error copying h_rand_vec_init to device"<<endl;
    }
    while(tol[0] > threshold && tol[1] > threshold && tol[2] > threshold && tol[3] > threshold && tol[4] > threshold && tol[5] > threshold){
        for(int i = 0; i < NUMSTRM; i++){
            //Initialize the random vector
            if(tol[i] > threshold){
                Init_Random<<<blocks_per_stream, TPB,0,streams[i]>>>(rand_vec_init, rand_init, node_size, K);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    cout<<"Error synchronizing device at Init Random for Stream "<<i<<endl;
                }
            }
        }
        for(int i = 0; i < NUMSTRM; i++){
            //Perform the first iteration of the algorithm
            if(tol[i] > threshold){
                sparseCSRMat_Vec_Mult<<<blocks_per_stream, TPB,0,streams[i]>>>(d_csc, d_succ, d_values, rand_vec_init, d_res, node_size);  
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    cout<<"Error synchronizing device at sparseCSRMat_Vec_Mult for stream "<<i<<endl;
                }
            }
        }
            // Just Verify that the multiplication is working
        for(int i = 0; i < NUMSTRM; i++){
            // Add 1/n to the vector
            if(tol[i] > threshold){
                Float_VectAdd<<<blocks_per_stream, TPB,0,streams[i]>>>(d_vec, rand_vec_init, node_size);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    cout<<"Error synchronizing device for Float_VectAdd at stream "<<i<<endl;
                }
                if(!HandleCUDAError(cudaMemcpy(rand_vec_init, d_vec, sizeof(float)*node_size, cudaMemcpyDeviceToDevice))){
                    cout<<"Error copying d_vec to host"<<endl;
                }
                //Need to normalize the vector using thrust library

                l2_norm_d_vec[i] = thrust::transform_reduce(thrust::device, d_vec, d_vec + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                l2_norm_d_vec[i] = sqrt(l2_norm_d_vec[i]);

                l2_norm_rand_vec_init[i] = thrust::transform_reduce(thrust::device, rand_vec_init, rand_vec_init + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
                l2_norm_rand_vec_init[i] = sqrt(l2_norm_rand_vec_init[i]);


                sum[i] = thrust::reduce(thrust::device.on(streams[i]), rand_vec_init, rand_vec_init+node_size);
                thrust::transform(thrust::device.on(streams[i]), rand_vec_init, rand_vec_init+node_size, rand_vec_init, thrust::placeholders::_1/sum);
                thrust::fill(thrust::device.on(streams[i]), d_vec, d_vec+node_size, 1.0f/node_size);
                if(!HandleCUDAError(cudaStreamSynchronize(streams[i]))){
                    cout<<"Error synchronizing device"<<endl;
                }
            }
        }
    }
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
            cout<<"Error creating stream number "<<i<<endl;
        }
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*(node_size+1)))){
        cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size)))){
        cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size))){
        cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_res, sizeof(float)*node_size))){
        cout<<"Error allocating memory for d_res"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_vec, vec, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying vec to device"<<endl;
    }
    delete[] vec;
    if(!HandleCUDAError(cudaMemcpy(d_res, res, sizeof(float)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying res to device"<<endl;
    }
    delete[] res;
    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size)))){
        cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_values, values, sizeof(float)*edge_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying values to device"<<endl;
    }
    // delete[] values;
    // if(!HandleCUDAError(cudaMemcpy(values,d_values, sizeof(float)*edge_size, cudaMemcpyDeviceToHost))){
    //     cout<<"Error copying values to device"<<endl;
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
        cout<<"Error copying h_rand_vec_init to device"<<endl;
    }
    //Initialize the random vector
    Init_Random<<<blocks_per_stream, TPB>>>(rand_vec_init, rand_init, node_size, K);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    //Perform the first iteration of the algorithm
    sparseCSRMat_Vec_Mult<<<blocks_per_stream, TPB>>>(d_csc, d_succ, d_values, rand_vec_init, d_res, node_size);  
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    Float_VectAdd<<<blocks_per_stream, TPB>>>(d_res, d_vec, node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_rand_vec_init, rand_vec_init, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        cout<<"Error copying d_vec to host"<<endl;
    }
    float* h_res_GPU = new float[node_size]{0.0f};

    if(!HandleCUDAError(cudaMemcpy(h_res_GPU, d_res, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        cout<<"Error copying d_vec to host"<<endl;
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
        cout<<"Error synchronizing device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_res_GPU, d_res, sizeof(float)*node_size, cudaMemcpyDeviceToHost))){
        cout<<"Error copying d_vec to host"<<endl;
    }
    Normalize_L2(h_res_CPU, node_size);
    Verify(h_res_GPU, h_res_CPU, node_size);

}

__global__ void sparseCSRMat_Vec_Mult(unsigned int* csc, unsigned int* succ, float* values, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        unsigned int start = csc[t];
        unsigned int end = csc[t+1];
        float sum = 0.0f;
        for(int i = start; i < end; i++){
            sum += values[i]*vec[succ[i]];
        }
        result[t] = sum;
    }
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
            cout<<"Error at index "<<i<<endl;
            cout<<"GPU: "<<gpu_vec[i]<<endl;
            cout<<"CPU: "<<cpu_vec[i]<<endl;
            return;
        }
    }
    cout<<"No errors found"<<endl;
}