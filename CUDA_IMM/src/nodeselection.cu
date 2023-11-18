#include "../include/IMM.h"
/*
The psuedocode for the node selection is as follows
1) Initialize a Node set = {empty set}
2) for i =1 to k do:
3)    Identify the node thaat maximizes Fr({S} U {v})-Fr({S})
4)    Add the node to the set
5) return the set


*/


//This is likely an unoptimized version , but will take in a set of nodes in an RRR set, and returns a count
__global__ void NodeSel_V1(unsigned int* hist_bin, unsigned int* RR_nodes, unsigned int size){
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //Get the thread coordinates
    // Can we use shared memory?
    for(int i=idx;i<size;i+=blockDim.x*gridDim.x){
        hist_bin[RR_nodes[i]]+=1;
    }

}