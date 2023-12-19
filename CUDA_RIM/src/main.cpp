#include "../include/data.h"

#include <string.h>


#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <unistd.h>


int main(int argc, char** argv)
{
    if(argc < 2){
        cout << "Please specify the data set" << endl;
        exit(0);
    }
    else{
        if(strcmp(argv[1], "HT") == 0){
            unsigned int no_nodes, no_edges;
            get_graph_info(HEPTH_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(HEPTH_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR(edge_list,csr,succ,no_nodes,no_edges);
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_PR);
            else if(strcmp(argv[3],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }
            Export_Seed_Set_to_CSV(seed_set,K,HEPTH_SEED_PATH);
        }
        else if(strcmp(argv[1],"WK")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(WIKI_VOTE_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(WIKI_VOTE_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR(edge_list,csr,succ,no_nodes,no_edges);
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_PR);
            else if(strcmp(argv[3],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }
            Export_Seed_Set_to_CSV(seed_set,K,WIKI_VOTE_SEED_PATH);
        }
        else if(strcmp(argv[1],"AR")==0){
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
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_PR);
            else if(strcmp(argv[3],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }
            Export_Seed_Set_to_CSV(seed_set,K,ARVIX_SEED_PATH);
        }
        else if(strcmp(argv[1],"PLN")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(HOMO_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(HOMO_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_PR); 
            else if(strcmp(argv[3],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,SEED_PATH);
        }
        else{
            cout << "Please specify the data set" << endl;
            exit(0);
        }
    return 0;
    }
}