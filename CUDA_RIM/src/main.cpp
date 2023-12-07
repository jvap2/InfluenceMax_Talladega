#include "../include/data.h"

#include <string.h>


#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <unistd.h>


int main(int argc, char** argv)
{
    //read in the data, for IC
    if(argc < 2){
        cout << "Please specify the diffusion model" << endl;
        exit(0);
    }
    else{
        if(strcmp(argv[1], "IC") == 0){
            unsigned int no_nodes, no_edges;
            get_graph_info(ARVIX_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(ARVIX_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR(edge_list,csr,succ,no_nodes,no_edges);
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set);
            Export_Seed_Set_to_CSV(seed_set,K,ARVIX_SEED_PATH);
        }
        else if(strcmp(argv[1],"LT")==0){
            cout<<"Incomplete, come back later"<<endl;
            exit(0);
        }
    }

    return 0;
}