#include "../include/IMM.h"
#include "../include/data.h"

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
            IMM_Ver1();
        }
        else if(strcmp(argv[1],"LT")==0){
            cout<<"Incomplete, come back later"<<endl;
            exit(0);
        }
    }

    return 0;
}