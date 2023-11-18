#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>   
#include<algorithm>
#include<cstdlib>
#include<ctime>
using namespace std;
#include "IMM.h"

#define homo_path "../Graph_Data_Storage/homo.csv"
#define homo_data_path "../Graph_Data_Storage/homo_info.csv"

void readData(string filename, edge_t_IC* edge_list);

void get_graph_info(string path, unsigned int* nodes, unsigned int* edges);

void genCSC_LT(edge_t_LT* edge_list, node_t_LT* succ, unsigned int* csc, unsigned int node_size, unsigned int edge_size);

void genCSC_IC(edge_t_IC* edge_list, unsigned int* succ, unsigned int* csc, unsigned int node_size, unsigned int edge_size);