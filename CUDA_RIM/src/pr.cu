#include "../include/data.h"
#include "../include/GPUErrors.h"


/*We are going to simulate the same function as the nvgraph page rank
it utilizes cublas and cusparse*/


__global__ void Init_P(float* P, unsigned int node_size, float* damp){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < node_size && y < node_size){
            P[y*node_size + x] = (*damp/(1.0*node_size));
    }
}




__global__ void Gen_P_Mem_eff(float* weight_P, unsigned int* src, unsigned int* succ, unsigned int node_size, float* damp){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<node_size){
        //We need to find a node in the src
        //We will then iterate through the succ, and access the src[succ+1] to src[succ] to 
        // get the out degree
        unsigned int num_succ=src[idx+1]-src[idx];
        if(num_succ!=0){
            for(unsigned int i=src[idx]; i<src[idx+1]; i++){
                unsigned int succ_node = succ[i];//Get the node number of the successor
                weight_P[succ_node*node_size+idx]+=(1.0-*damp)/(1.0f*num_succ);
            }
        }
    }
}


__global__ void Init_P_Sparse(float* weight_P,const int* src,const int* succ, unsigned int node_size, float* damp){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<node_size){
        //We need to find a node in the src
        //We will then iterate through the succ, and access the src[succ+1] to src[succ] to 
        // get the out degree
        unsigned int num_succ=src[idx+1]-src[idx];
        if(num_succ!=0){
            for(unsigned int i=src[idx]; i<src[idx+1]; i++){
                unsigned int succ_node = succ[i];//Get the node number of the successor
                weight_P[i]=(1.0-*damp)/(1.0f*num_succ);
            }
        }
    }
}

__global__ void Init_Pr(float* pr_vector, unsigned int node_size){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < node_size){
        pr_vector[x] = 1.0f/node_size;
    }
}

__host__ void PageRank(float* pr_vector, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time){
    float alpha = 1.0; 
    float beta = 0.0;
    float tol_temp=100.0f;
    float* d_P;
    unsigned int* d_global_src;
    unsigned int* d_global_succ;
    float* d_pr_vector;
    float* dr_pr_vector_temp;
    float* d_damp;
    float norm=0;
    float norm_temp=0;
    unsigned int tpb = 256;
    unsigned int blocks = (node_size+tpb-1)/tpb;
    unsigned int tpb_2d = 16;
    unsigned int blocks_2d = (node_size+tpb_2d-1)/tpb_2d;
    dim3 Threads(tpb_2d, tpb_2d);
    dim3 Blocks(blocks_2d, blocks_2d);
    unsigned int blocks_edge = (edge_size+tpb-1)/tpb;
    if(!HandleCUDAError(cudaMalloc((void**)&d_P, node_size*node_size*sizeof(float)))){
        cout<<"Error allocating memory for P"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for global_src"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, edge_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for global_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_global_src, global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying global_src to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, edge_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying global_succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_damp, sizeof(float)))){
        cout<<"Error allocating memory for damp"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_damp, &damp, sizeof(float), cudaMemcpyHostToDevice))){
        cout<<"Error copying damp to device"<<endl;
    }
    cout<< "2D grid dim: "<<Blocks.x<<" "<<Blocks.y<<endl;
    cout<< "2D block dim: "<<Threads.x<<" "<<Threads.y<<endl;
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
                static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }


    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("  Total amount of constant memory:               %zu bytes\n",
            deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
            deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
            deviceProp.sharedMemPerMultiprocessor);
    }


    size_t free_byte ;
    size_t total_byte ;
    if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
        cout<<"Error getting memory info"<<endl;
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    Init_P<<<Blocks,Threads>>>(d_P, node_size, d_damp);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device with Initializing P"<<endl;
    }
    Gen_P_Mem_eff<<<blocks,tpb>>>(d_P, d_global_src, d_global_succ, node_size, d_damp);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device with Generating P"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_global_src))){
        cout<<"Error freeing memory for global_src"<<endl;
    }
    
    //Now, we need to initialize the pr_vector
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr_vector, node_size*sizeof(float)))){
        cout<<"Error allocating memory for pr_vector"<<endl;
    }
    Init_Pr<<<blocks,tpb>>>(d_pr_vector, node_size);
    if(!HandleCUDAError(cudaMalloc((void**)&dr_pr_vector_temp, node_size*sizeof(float)))){
        cout<<"Error allocating memory for dr_pr_vector_temp"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(dr_pr_vector_temp, 0, node_size*sizeof(float)))){
        cout<<"Error setting dr_pr_vector_temp to 0"<<endl;
    }
    cublasHandle_t handle;
    if(!HandleCUBLASError(cublasCreate(&handle))){
        cout<<"Error creating cublas handle"<<endl;
    }
    cout<<"Performing PageRank"<<endl;
    unsigned int iter_temp=max_iter;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    while(max_iter>0 && tol_temp>tol){
        cublasSgemv_v2(handle, CUBLAS_OP_T, node_size, node_size, &alpha, d_P, node_size, d_pr_vector, 1, &beta, dr_pr_vector_temp, 1);
        cublasSnrm2_v2(handle, node_size, dr_pr_vector_temp, 1, &norm_temp);
        cublasSnrm2_v2(handle, node_size, d_pr_vector, 1, &norm);
        tol_temp = fabsf(norm_temp-norm);
        cublasScopy_v2(handle, node_size, dr_pr_vector_temp, 1, d_pr_vector, 1);
        max_iter--;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Time elapsed PageRank: "<<milliseconds<<" ms"<<endl;
    cout<<"Converged in "<<iter_temp-max_iter<<" iterations"<<endl;
    cout<<"Tolerance: "<<tol_temp<<endl;
    *time=milliseconds;

    cout<<"PageRank finished"<<endl;
    if(!HandleCUDAError(cudaMemcpy(pr_vector, d_pr_vector, node_size*sizeof(float), cudaMemcpyDeviceToHost))){
        cout<<"Error copying pr_vector to host"<<endl;
    }
    cout<<"Copied pr_vector to host"<<endl;
    if(!HandleCUDAError(cudaFree(d_P))){
        cout<<"Error freeing memory for P"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_global_succ))){
        cout<<"Error freeing memory for global_succ"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_pr_vector))){
        cout<<"Error freeing memory for pr_vector"<<endl;
    }
    if(!HandleCUDAError(cudaFree(dr_pr_vector_temp))){
        cout<<"Error freeing memory for dr_pr_vector_temp"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_damp))){
        cout<<"Error freeing memory for damp"<<endl;
    }
    if(!HandleCUBLASError(cublasDestroy(handle))){
        cout<<"Error destroying cublas handle"<<endl;
    }
}




__host__ void PageRank_Sparse(float* pr_vector, int* global_csc, int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol, float* time){
    float tol_temp=100.0f;
    float* d_P;
    int* d_global_src;
    int* d_global_succ;
    float* d_pr_vector;
    float* dr_pr_vector_temp;
    float* d_damp;
    float* d_support_vect;
    float norm=0;
    float norm_temp=0;
    unsigned int tpb = 256;
    unsigned int blocks_node = (node_size+tpb-1)/tpb;
    unsigned int blocks_edge = (edge_size+tpb-1)/tpb;
    if(!HandleCUDAError(cudaMalloc((void**)&d_P, edge_size*sizeof(float)))){
        cout<<"Error allocating memory for P"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(int)))){
        cout<<"Error allocating memory for global_src"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, edge_size*sizeof(int)))){
        cout<<"Error allocating memory for global_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_global_src, global_csc, (node_size+1)*sizeof(int), cudaMemcpyHostToDevice))){
        cout<<"Error copying global_src to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, edge_size*sizeof(int), cudaMemcpyHostToDevice))){
        cout<<"Error copying global_succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_damp, sizeof(float)))){
        cout<<"Error allocating memory for damp"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_damp, &damp, sizeof(float), cudaMemcpyHostToDevice))){
        cout<<"Error copying damp to device"<<endl;
    }
    Init_P_Sparse<<<blocks_edge,tpb>>>(d_P, d_global_src, d_global_succ, node_size, d_damp);
    //Now we need to take the transpose of the csr format of this using cusparse
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    if(!HandleCUSparseError(cusparseCreate(&handle))){
        cout<<"Error creating cusparse handle"<<endl;
    }
    if(!HandleCUSparseError(cusparseCreateMatDescr(&descr))){
        cout<<"Error creating cusparse matrix descriptor"<<endl;
    }
    if(!HandleCUSparseError(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL))){
        cout<<"Error setting cusparse matrix type"<<endl;
    }
    if(!HandleCUSparseError(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO))){
        cout<<"Error setting cusparse matrix index base"<<endl;
    }
    int* d_csr_row_ptr;
    int* d_csr_col_ind;
    float* d_csr_val;
    void *buffer = NULL;
    size_t bufferSize = 0;
    if(!HandleCUDAError(cudaMalloc((void**)&d_csr_row_ptr, (node_size+1)*sizeof(int)))){
        cout<<"Error allocating memory for csr_row_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_csr_col_ind, edge_size*sizeof(int)))){
        cout<<"Error allocating memory for csr_col_ind"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_csr_val, edge_size*sizeof(float)))){
        cout<<"Error allocating memory for csr_val"<<endl;
    }
    if(!HandleCUSparseError(cusparseCsr2cscEx2_bufferSize(handle, node_size, node_size, edge_size, d_P, d_global_src, d_global_succ, d_csr_val, d_csr_col_ind, d_csr_row_ptr,CUDA_R_32F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,&bufferSize))){
        cout<<"Error getting buffer size for csr2csc"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc(&buffer, bufferSize))){
        cout<<"Error allocating memory for buffer"<<endl;
    }
    if(!HandleCUSparseError(cusparseCsr2cscEx2(handle, node_size, node_size, edge_size, d_P, d_global_src, d_global_succ, d_csr_val, d_csr_col_ind, d_csr_row_ptr,CUDA_R_32F,CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,buffer))){
        cout<<"Error converting csr to csc"<<endl;
    }
    //Now we need to get the transpose of the csr matrix
    if(!HandleCUDAError(cudaFree(d_P))){
        cout<<"Error freeing memory for P"<<endl;
    }
    if(!HandleCUDAError(cudaFree(const_cast<int*>(d_global_succ)))){
        cout<<"Error freeing memory for global_succ"<<endl;
    }
    if(!HandleCUDAError(cudaFree(const_cast<int*>(d_global_src)))){
        cout<<"Error freeing memory for global_src"<<endl;
    }
    //Now, we need to initialize the pr_vector and support vector
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr_vector, node_size*sizeof(float)))){
        cout<<"Error allocating memory for pr_vector"<<endl;
    }
    thrust::fill(thrust::device, d_pr_vector, d_pr_vector+node_size, 1.0f/node_size);
    if(!HandleCUDAError(cudaMalloc((void**)&dr_pr_vector_temp, node_size*sizeof(float)))){
        cout<<"Error allocating memory for dr_pr_vector_temp"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(dr_pr_vector_temp, 0, node_size*sizeof(float)))){
        cout<<"Error setting dr_pr_vector_temp to 0"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_support_vect, node_size*sizeof(float)))){
        cout<<"Error allocating memory for support vector"<<endl;
    }
    thrust::fill(thrust::device, d_support_vect, d_support_vect+node_size, 1.0f/node_size);
    cout<<"Performing PageRank"<<endl;
    unsigned int iter_temp=max_iter;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float l_2_pr=0;
    float l_2_pr_temp=0;
    float sum=0;
    float old_tol=10;
    while(max_iter>0 && tol_temp>tol){
        sparseCSRMat_Vec_Mult<int><<<blocks_node,tpb>>>(d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_pr_vector, dr_pr_vector_temp, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device with sparse matrix vector multiplication"<<endl;
        }
        Float_VectAdd<<<blocks_node,tpb>>>(dr_pr_vector_temp, d_support_vect, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device with vector addition"<<endl;
        }
        //Use thrust to normalize the pr_vector
        // l_2_pr = thrust::transform_reduce(thrust::device, d_pr_vector, d_pr_vector + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
        // l_2_pr = sqrt(l_2_pr);

        l_2_pr_temp = thrust::transform_reduce(thrust::device, dr_pr_vector_temp, dr_pr_vector_temp + node_size, [] __device__ (float x) { return x * x; }, 0.0f, thrust::plus<float>());
        l_2_pr_temp = sqrt(l_2_pr_temp);

        tol_temp = abs(old_tol-l_2_pr_temp);
        old_tol=l_2_pr_temp;
        thrust::copy(thrust::device, dr_pr_vector_temp, dr_pr_vector_temp+node_size, d_pr_vector);
        thrust::fill(thrust::device, dr_pr_vector_temp, dr_pr_vector_temp+node_size, 0.0f);


        sum = thrust::reduce(thrust::device, d_pr_vector, d_pr_vector+node_size);
        float temp = sum;
        thrust::transform(thrust::device, d_pr_vector, d_pr_vector+node_size, d_pr_vector, [=] __device__ (float x) { return x/temp; });
        max_iter--;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *time=milliseconds;
    if(!HandleCUDAError(cudaFree(d_csr_row_ptr))){
        cout<<"Error freeing memory for csr_row_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_csr_col_ind))){
        cout<<"Error freeing memory for csr_col_ind"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_csr_val))){
        cout<<"Error freeing memory for csr_val"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(pr_vector, d_pr_vector, node_size*sizeof(float), cudaMemcpyDeviceToHost))){
        cout<<"Error copying pr_vector to host"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_pr_vector))){
        cout<<"Error freeing memory for pr_vector"<<endl;
    }
    if(!HandleCUDAError(cudaFree(dr_pr_vector_temp))){
        cout<<"Error freeing memory for dr_pr_vector_temp"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_support_vect))){
        cout<<"Error freeing memory for support vector"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_damp))){
        cout<<"Error freeing memory for damp"<<endl;
    }
    if(!HandleCUSparseError(cusparseDestroy(handle))){
        cout<<"Error destroying cusparse handle"<<endl;
    }

}