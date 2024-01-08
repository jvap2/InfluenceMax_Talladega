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
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,HEPTH_DATA_MEASURE_PR,HEPTH_PR);
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_GREEDY,HEPTH_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,HEPTH_DATA_MEASURE_SIG,HEPTH_PR);
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
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,WIKI_VOTE_DATA_MEASURE_PR, WIKI_VOTE_PR);
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_GREEDY,WIKI_VOTE_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,WIKI_VOTE_DATA_MEASURE_SIG,WIKI_VOTE_PR);
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
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,ARVIX_DATA_MEASURE_PR,ARVIX_PR);
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_GREEDY,ARVIX_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,ARVIX_DATA_MEASURE_SIG,ARVIX_PR);
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
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,HOMO_DATA_MEASURE_PR,HOMO_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_GREEDY,HOMO_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,HOMO_DATA_MEASURE_SIG,HOMO_PR);
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
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,EP_DATA_MEASURE_PR,EP_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE_GREEDY,EP_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,EP_DATA_MEASURE_SIG,EP_PR);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,EP_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1],"AM")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(AM_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(AM_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,AM_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,AM_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,AM_DATA_MEASURE_PR,AM_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,AM_DATA_MEASURE_GREEDY,AM_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,AM_DATA_MEASURE_SIG,AM_PR);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,AM_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1],"ND")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(ND_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(ND_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,ND_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,ND_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,ND_DATA_MEASURE_PR,ND_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,ND_DATA_MEASURE_GREEDY,ND_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,ND_DATA_MEASURE_SIG,ND_PR);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,ND_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1],"GGL")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(GGL_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(GGL_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,GGL_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,GGL_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,GGL_DATA_MEASURE_PR,GGL_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,GGL_DATA_MEASURE_GREEDY,GGL_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,GGL_DATA_MEASURE_SIG,GGL_PR);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,GGL_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1],"BRK")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(BRK_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(BRK_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,BRK_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,BRK_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,BRK_DATA_MEASURE_PR,BRK_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,BRK_DATA_MEASURE_GREEDY,BRK_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,BRK_DATA_MEASURE_SIG,BRK_PR);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,BRK_SEED_PATH);

            delete[] csr;
            delete[] succ;
            free(edge_list);
        }
        else if(strcmp(argv[1],"WKT")==0){
            unsigned int no_nodes, no_edges;
            get_graph_info(WKT_DATA_PATH, &no_nodes, &no_edges);
            cout << "no_nodes: " << no_nodes << endl;
            cout << "no_edges: " << no_edges << endl;
            edge* edge_list= (edge*)malloc(sizeof(edge)*no_edges);
            readData(WKT_PATH,edge_list);
            unsigned int *csr, *succ;
            csr = new unsigned int[no_nodes+1];
            succ = new unsigned int[no_edges];
            thrust::fill(succ,succ+no_edges,0);
            thrust::fill(csr,csr+no_nodes+1,0);
            genCSR<unsigned int>(edge_list,csr,succ,no_nodes,no_edges);
            cout<<"Generated CSR"<<endl;
            unsigned int* seed_set = new unsigned int[K];
            // CheckSparseMatVec(csr,succ,edge_list,no_nodes,no_edges);
            if(strcmp(argv[2],"one")==0)
                RIM_rand_Ver1(csr,succ,no_nodes,no_edges,seed_set,WKT_DATA_MEASURE);
            else if(strcmp(argv[2],"two")==0)
                RIM_rand_Ver2(csr,succ,no_nodes,no_edges,seed_set,WKT_DATA_MEASURE_2);
            else if(strcmp(argv[2],"pr")==0)
                RIM_rand_Ver3_PR(csr,succ,no_nodes,no_edges,seed_set, edge_list,WKT_DATA_MEASURE_PR,WKT_PR); 
            else if(strcmp(argv[2],"greedy")==0)
                RIM_rand_Ver4_Greedy(csr,succ,no_nodes,no_edges,seed_set,WKT_DATA_MEASURE_GREEDY,WKT_PR);
            else if(strcmp(argv[2],"sigmoid")==0)
                RIM_rand_Ver5_Sig(csr,succ,no_nodes,no_edges,seed_set,WKT_DATA_MEASURE_SIG,WKT_PR);
            else{
                cout << "Please specify the version" << endl;
                exit(0);
            }        
            Export_Seed_Set_to_CSV(seed_set,K,WKT_SEED_PATH);

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
            PageRank_Sparse(pr_vector,.15f,no_nodes,no_edges, 100, 1e-6,time_2,ARVIX_PR);
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