#include "../include/GPUErrors.h"

bool HandleCUDAError(cudaError_t t)
{
	if (t != cudaSuccess)
	{
		cout << cudaGetErrorString(cudaGetLastError())<<endl;//This will get the string of the error for blocking error
		cout<<t<<endl;
		return false;
	}
	return true;
}
//We can have runtime errors on the GPU, which is what the function below is used for
bool GetCUDARunTimeError()
{
	cudaError_t t = cudaGetLastError();
	if (t != cudaSuccess)
	{
		cout << cudaGetErrorString(t) << endl;
		return false;
	}
	return true;
}

bool HandleCUSparseError(cusparseStatus_t t){
	if (t != CUSPARSE_STATUS_SUCCESS)
	{
		cout << "CUSPARSE ERROR: " << t << endl;
		cout<< cusparseGetErrorString(t)<<endl;
		return false;
	}
	return true;
}

bool HandleCUBLASError(cublasStatus_t t){
	if (t != CUBLAS_STATUS_SUCCESS)
	{
		cout << "CUBLAS ERROR: " << t << endl;
		cout<< cublasGetStatusString(t)<<endl;
		return false;
	}
	return true;
}


void printCudaMemoryUsage() {
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %.2f, free = %.2f MB, total = %.2f MB\n",
        used_db /1024.0/1024.0, free_db /1024.0/1024.0, total_db /1024.0/1024.0);
}