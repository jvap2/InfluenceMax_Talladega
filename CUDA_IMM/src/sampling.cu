#include "../include/IMM.h"
#include "../include/data.h"
/*The code for this come from Tang et al IMM Algorithm
The psuedo code is as follows:
1) Initialize a set R={empty set} and an integer LB = 1
2) Let epsilon'=sqrt(2)epsilon
3) for i = 1 to log_2(n)-1 do:
4) 	Let x=n/2^i
5) 	Let theta = lambda'/x
6)  while |R|<theta_i do:
7)      Select a node from G uniformly at random
8)      Generate an RR set for v, and insert it into R
9)  Let S_i = NodeSelection(R)
10) if n*FR(S)>=(1+epsilon')*x then:
11)     LB = n*FR(S)/(1+epsilon')
12)     break
13) Let theta= lambda^{*}/LB
14) while |R|<theta do:
15)     Select a node from G uniformly at random
16)     Generate an RR set for v, and insert it into R
17) Return R


What is going to be needed:
1) A graph
2) A set of RR sets
3) A set of nodes
4) A set of edges

How do we want to store these:
1) The RR sets would be traversable with CSC format, and any forward traversal will need
but would it be convenient to traverse both ways with a COO format?
2) We may be able to use linked lists as well but this will be slower, but also harder to implement, and harder for generating RRR sets

*/


__host__ void IMM_Ver1(){
    //Step 1
    //Step 2
    unsigned int n, m;
    get_graph_info(homo_data_path,&n,&m);
    cout<<n<<endl;
    cout<<m<<endl;
    edge_t_IC* edge_list = new edge_t_IC[m];
    readData(homo_path,edge_list);
    unsigned int *succ, *csc;
    if(!HandleCUDAError(cudaMallocManaged((void**)&succ,m*sizeof(unsigned int),cudaMemAttachHost))){
        cout<<"Unable to allocate reverse successors"<<endl;
    }
    if(!HandleCUDAError(cudaMallocManaged((void**)&csc,(n+1)*sizeof(unsigned int),cudaMemAttachHost))){
        cout<<"Unable to allocate csc vertices"<<endl;
    }
    genCSC_IC(edge_list,succ,csc,n,m);
    delete[] edge_list;
    int l=1;
    float epsilon = .5;
    float epsilon_prime = sqrt(2)*epsilon;
    //Step 3
    int num_iterations =(int)(log2(n)-1);
    unsigned int i = 1;
    unsigned int x = n/2;
    float nck= Calc_LogComb(n,K); 
    float lambda_prime=(2+(2.0f/3.0f)*epsilon_prime)*(nck+l*logf(n)+logf(log2f(n)))*n/(powf(epsilon_prime,2));
    float alpha = sqrtf(l*logf(n)+logf(2));
    float beta = sqrtf((1-1/expf(1))*(nck+l*logf(n)+logf(2)));
    float lambda_star = 2*n*(powf((1-expf(-1))*(alpha+beta),2.0f)/(epsilon*epsilon));
    float theta = lambda_prime/x;
    cout<<nck<<endl;
    cout<<theta<<endl;
    //Step 4
    //Now we need
}

__host__ float Calc_LogComb(unsigned int n, unsigned int k){
    float temp_1=0;
    float temp_2=0;
    for(unsigned int i = 0; i<n-k;i++){
        temp_1+=logf(k+i+1);
        temp_2+=logf(i+1);
    }
    return temp_1-temp_2;
}