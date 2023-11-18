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

void readData(string filename, edge_t_IC* edge_list);

void genCSC_LT(edge_t_IC* edge_list, node_t_LT* succ, unsigned int* csc, unsigned int size);

void genCSC_LT(edge_t_IC* edge_list, node_t_LT* succ, unsigned int* csc, unsigned int size);