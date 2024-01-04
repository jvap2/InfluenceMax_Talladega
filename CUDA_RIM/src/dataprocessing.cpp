#include "../include/data.h"


void readData(string filename, edge* edge_list){
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

void genCSC(edge* edge_list, unsigned int* succ, unsigned int* csc, unsigned int node_size, unsigned int edge_size){
    unsigned int* csc_temp = new unsigned int[node_size+1]{0};
    for(unsigned int i = 0; i < edge_size; i++){
        csc_temp[edge_list[i].dst]++;
        succ[i] = edge_list[i].src;
    }
    csc[0] = 0;
    for(unsigned int i = 1; i<=node_size; i++){
        csc[i] = csc[i-1] + csc_temp[i-1];
    }
    delete[] csc_temp;
}





void GenAdj(edge* edge_list, float* adj, unsigned int node_size, unsigned int edge_size){
    for(int i=0; i<edge_size;i++){
        adj[edge_list[i].src*node_size+edge_list[i].dst]=1.0f;
    }
}

void h_MatVecMult(float* h_A, float* h_x, float* h_y, unsigned int node_size){
    for(int i=0; i<node_size;i++){
        float temp=0.0f;
        for(int j=0; j<node_size;j++){
            temp+=h_A[i*node_size+j]*h_x[j];
        }
        h_y[i]=temp;
    }
}

void Normalize_L2(float* h_x, unsigned int node_size){
    float norm=0.0f;
    for(int i=0; i<node_size;i++){
        norm+=h_x[i]*h_x[i];
    }
    norm=sqrt(norm);
    cout<<"CPU Norm: "<<norm<<endl;
    for(int i=0; i<node_size;i++){
        h_x[i]=h_x[i]/norm;
    }
}


void Export_Seed_Set_to_CSV(unsigned int* seed_set, unsigned int seed_size, string path){
    ofstream data;
    data.open(path);
    if(!data.is_open()){
        cout<<"Cannot open file"<<endl;
    }
    if(data.is_open()){
        data<<"Seed_Set"<<endl;
        for(int i=0; i<seed_size;i++){
            data<<seed_set[i]<<endl;
        }
    }
    data.close();
}

__host__ void Gen_Pr_Sprs(unsigned int* csc, unsigned int* succ, float* weight_P, unsigned int node_size, unsigned int edge_size, float damp, string file){
    ifstream data;
    data.open(file);
    string line,word;
    int count =0;
    int column = 0;
    unsigned int edge_list_src, edge_list_dst;
    edge_list_src = 0;
    edge_list_dst = 0;
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
                        edge_list_src=stoul(word);
                        csc[edge_list_src]++;
                        column++;
                    }
                    else if(column==1){
                        edge_list_dst=stoul(word);
                        succ[count-1]=edge_list_dst;
                        column++;
                    }
                    else{
                        weight_P[count-1]=(1-damp)*stof(word);
                        column++;
                    }
                }
                //Extract data until ',' is found
            }
            count++;
            column = 0;
        }
    }
    data.close();
    //Now, we need to prefix sum the src_ptr
    unsigned int* src_temp = new unsigned int[node_size+1]{0};
    for(int i=1; i<=node_size;i++){
        src_temp[i]=src_temp[i-1]+csc[i-1];
    }
    copy(src_temp, src_temp+node_size+1, csc);
}