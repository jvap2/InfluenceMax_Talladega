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
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*NUMSTRM);
    for(int i = 0; i < NUMSTRM; i++){
        if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
            cout<<"Error creating stream number "<<i<<endl;
        }
    }
    unsigned int* d_csc;
    unsigned int* d_succ;
    float* d_vec; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*node_size))){
        cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*(edge_size+1)))){
        cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_vec, sizeof(float)*node_size))){
        cout<<"Error allocating memory for d_seed_set"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_csc, csc, sizeof(unsigned int)*node_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying csc to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_succ, succ, sizeof(unsigned int)*edge_size, cudaMemcpyHostToDevice))){
        cout<<"Error copying succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_vec, 1.0f/node_size, sizeof(float)*node_size))){
        cout<<"Error setting d_vec to 1/n"<<endl;
    }
    
    float* d_values;
    if(!HandleCUDAError(cudaMalloc((void**)&d_values, sizeof(float)*(edge_size+1)))){
        cout<<"Error allocating memory for d_values"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_values, 1.0f, sizeof(float)*(edge_size+1)))){
        cout<<"Error setting d_values to 1"<<endl;
    }
    unsigned int num_blocks = (node_size+TPB-1)/TPB;
    unsigned int num_blocks2 = (edge_size+TPB-1)/TPB;
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
    if(!HandleCUDAError(cudaMalloc((void**)&rand_vec_init, sizeof(float)*node_size))){
        std::cout<<"Error allocating memory for rand_vec_init"<<endl;
    } 
    sparseCSRMat_Vec_Mult<<<num_blocks2,TPB>>>(d_csc, d_succ, d_values,d_vec, node_size);


    
}


__global__ void sparseCSRMat_Vec_Mult(unsigned int* csc, unsigned int* succ, float* vec, float* result, unsigned int node_size){
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int t = tid; t < node_size; t+=blockDim.x*gridDim.x){
        unsigned int start = csc[t];
        unsigned int end = csc[t+1];
        unsigned int sum = 0;
        for(int i = start; i < end; i++){
            sum += vec[succ[i]];
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
    if(tid<k){
        unsigned int idx = (unsigned int)(rand_init[tid]*size);
        idx = idx%size;
        vec[idx] = 1.0f;
    }
}