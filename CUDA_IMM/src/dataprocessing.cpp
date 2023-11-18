#include "../include/data.h"


void readData(string filename, unsigned int* src, unsigned int* dst){
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
        src[i] = a;
        dst[i] = b;
        i++;
    }
}