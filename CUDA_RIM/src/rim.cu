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
    unsigned int* d_seed_set; //we will use the seed set as the PR vector and then transfer the top k to the actual seed set
    if(!HandleCUDAError(cudaMalloc((void**)&d_csc, sizeof(unsigned int)*node_size))){
        cout<<"Error allocating memory for d_csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, sizeof(unsigned int)*edge_size))){
        cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_seed_set, sizeof(unsigned int)*node_size))){
        cout<<"Error allocating memory for d_seed_set"<<endl;
    }

    
}