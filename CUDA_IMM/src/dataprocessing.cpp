#include "../include/data.h"


void readData(string filename, edge_t_IC* edge_list){
    ifstream file(filename);
    string line;
    int i = 0;
    unsigned int line_count=0;
    while (getline(file, line))
    {
        if(line_count==0){
            line_count++;
            continue;
        }
        istringstream iss(line);
        unsigned int a, b;
        if (!(iss >> a >> b)) { break; } // error
        edge_list[i].src = a;
        edge_list[i].dst = b;
        i++;
    }
}

void genCSC_LT(edge_t_LT* edge_list, node_t_LT* succ, unsigned int* csc, unsigned int size){
    unsigned int* csc_temp = new unsigned int[size]{0};
    for(unsigned int i = 0; i < size; i++){
        csc_temp[edge_list[i].dst.num]++;
        succ[i] = edge_list[i].src;
    }
    csc[0] = 0;
    for(unsigned int i = 1; i<size; i++){
        csc[i] = csc[i-1] + csc_temp[i-1];
    }
}

void genCSC_IC(edge_t_IC* edge_list, unsigned int* succ, unsigned int* csc, unsigned int size){
    unsigned int* csc_temp = new unsigned int[size]{0};
    for(unsigned int i = 0; i < size; i++){
        csc_temp[edge_list[i].dst]++;
        succ[i] = edge_list[i].src;
    }
    csc[0] = 0;
    for(unsigned int i = 1; i<size; i++){
        csc[i] = csc[i-1] + csc_temp[i-1];
    }
}