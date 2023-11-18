#include "../include/IMM.h"

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
            get_graph_info(homo_data_path,&no_nodes,&no_edges);
            cout<<no_edges<<endl;
            edge_t_IC* edge_list = new edge_t_IC[no_edges];
            readData(homo_path,edge_list);
            unsigned int *succ, *csc;
            succ = new unsigned int[no_edges];
            csc = new unsigned int[no_nodes+1];
            genCSC_IC(edge_list,succ,csc,no_nodes,no_edges);
        }
        else if(strcmp(argv[1],"LT")==0){
            cout<<"Incomplete, come back later"<<endl;
            exit(0);
        }
    }

    return 0;
}