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
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,HEPTH_DATA_MEASURE_PR);
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }
            Export_Seed_Set_to_CSV(seed_set,K,HEPTH_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
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
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,WIKI_VOTE_DATA_MEASURE_PR);
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }
            Export_Seed_Set_to_CSV(seed_set,K,WIKI_VOTE_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
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
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,ARVIX_DATA_MEASURE_PR);
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }
            Export_Seed_Set_to_CSV(seed_set,K,ARVIX_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
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
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,HOMO_DATA_MEASURE_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1],"EP")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(EP_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(EP_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,EP_DATA_MEASURE_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE_GREEDY);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,EP_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1], "debug")==0){
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
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            int *pr_csc,*pr_succ;
            pr_csc = new int[no_nodes+1];
            pr_succ = new int[no_edges];
            thrust::fill(pr_succ,pr_succ+no_edges,0);
            thrust::fill(pr_csc,pr_csc+no_nodes+1,0);
            genCSR<int>(edge_list,pr_csc,pr_succ,no_nodes,no_edges);
            float* pr_full = new float[no_nodes];
            thrust::fill(pr_full,pr_full+no_nodes,0.0f);
            float* pr_vector = new float[no_nodes];
            thrust::fill(pr_vector,pr_vector+no_nodes,1.0f/no_nodes);
            float* time_1 = new float[1];
            float* time_2 = new float[1];
            time_1[0]=0.0f;
            time_2[0]=0.0f;
            PageRank(pr_full, csr,succ, .15f, no_nodes, no_edges, 100, 1e-6, time_1);
            PageRank_Sparse(pr_vector,pr_csc,pr_succ,.15f,no_nodes,no_edges, 100, 1e-6,time_2);
            Verify_Pr(pr_full,pr_vector,no_nodes);
            cout<<"Generated CSR"<<endl;
            // unsigned int* seed_set = new unsigned int[no_nodes];
            unsigned int* seed_set = new unsigned int[K];
        }
        else{
            cout << "Please specify the data set" << endl;
            exit(0);
        }
    return 0;
    }
}