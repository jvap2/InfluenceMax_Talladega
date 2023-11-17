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
