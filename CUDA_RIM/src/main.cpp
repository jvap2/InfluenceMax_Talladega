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
            get_graph_info(HOMO_DATA_PATH, &no_nodes, &no_edges);
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(HOMO_PATH,edge_list);
            unsigned int *csc, *succ;
            csc = new unsigned int[no_nodes];
            succ = new unsigned int[no_edges];
            genCSC(edge_list,succ,csc,no_nodes,no_edges);
            unsigned int* seed_set = new unsigned int[K];
            RIM_rand_Ver1(csc,succ,no_nodes,no_edges,seed_set);
        }
        else if(strcmp(argv[1],"LT")==0){
            cout<<"Incomplete, come back later"<<endl;
            exit(0);
        }
    }

    return 0;
}