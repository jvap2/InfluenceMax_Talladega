#include "../include/data.h"


void readData(string filename, edge_t_IC* edge_list){
    ifstream data;
    data.open(filename);
    string line,word;
    int count =0;
    int column = 0;
    if(!data.is_open()){
        cout<<"Cannot open file"<<endl;
    }
    if(data.is_open()){
        //Check if data is open
        while(getline(data,line)){
            //Keep extracting data until a delimiter is found
            stringstream stream_data(line); 
            while(getline(stream_data,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        edge_list[count-1].src=stoul(word);
                        column++;
                    }
                    else{
                        edge_list[count-1].dst=stoul(word);
                    }
                }
                //Extract data until ',' is found
            }
            count++;
            column = 0;
        }
    }

}

void get_graph_info(string path, unsigned int* nodes, unsigned int* edges){
    ifstream data;
    data.open(path);
    string line,word;
    int count =0;
    int column = 0;
    if(!data.is_open()){
        cout<<"Cannot open file"<<endl;
    }
    if(data.is_open()){
        //Check if data is open
        while(getline(data,line)){
            //Keep extracting data until a delimiter is found
            stringstream stream_data(line); 
            while(getline(stream_data,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        *nodes=stoi(word);
                        column++;
                    }
                    else{
                        *edges=stoi(word);
                    }
                }
                //Extract data until ',' is found
            }
            count++;
            column = 0;
        }
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