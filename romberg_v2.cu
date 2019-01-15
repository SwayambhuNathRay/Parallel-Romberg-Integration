#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define warp_size 32
#define Hwarp_size 16
#define A 0
#define B 15

void checkCUDAError(const char* msg);

__host__ __device__ inline double f(double x)
{
	return exp(x)*sin(x);
}


__host__ __device__ inline unsigned int getFirstSetBitPos(int n)
{
   return log2((float)(n&-n))+1;
}


__global__ void romberg(double a, double b, int row_size, double *result)	//row_size<=25, preferably 14
{
	extern __shared__ double local_array[];
	double diff = (b-a)/gridDim.x, step;
	int max_eval = (1<<(row_size-1)),k;
	b = a + (blockIdx.x+1)*diff;
	a += blockIdx.x*diff;
	
	step = (b-a)/max_eval;
	
	double local_col[25];
	for(int i = 0; i < row_size; i++)
		local_col[i] = 0.0;
	if(!threadIdx.x)
	{
		k = blockDim.x;
		local_col[0] = f(a) + f(b);
	}
	else
		k = threadIdx.x;
	
	for(; k < max_eval; k += blockDim.x)
	{
		local_col[row_size - getFirstSetBitPos(k)] += 2.0*f(a + step*k);
	}
	for(int i = 0; i < row_size; i++)
	{
		local_array[row_size*threadIdx.x + i] = local_col[i];
	}
	__syncthreads();
	if(threadIdx.x < row_size)
	{
		double sum = 0.0;
		for(int i = threadIdx.x; i < blockDim.x*row_size; i+=row_size)
			sum += local_array[i];
		
		local_array[threadIdx.x] = sum;
	}
	
	if(!threadIdx.x)
	{
		double *romberg_table = local_col;
		romberg_table[0] = local_array[0];
		for(int k = 1; k < row_size; k++)
			romberg_table[k] = romberg_table[k-1] + local_array[k];
		for(int k = 0; k < row_size; k++)	
			romberg_table[k]*= (b-a)/(1<<(k+1));
		
		for(int col = 0 ; col < row_size-1 ; col++)
		{
			for(int row = row_size-1; row > col; row--)
			{
				romberg_table[row] = romberg_table[row] + (romberg_table[row] - romberg_table[row-1])/((1<<(2*col+1))-1);
			}
		}
		result[blockIdx.x] = romberg_table[row_size-1];
	}

}


int main( int argc, char** argv)
{
	double *d_result, *h_result,sum=0.0;
	int numBlocks = 128, numThreadsPerBlock = 64, row_size = 13, max_eval = (1<<(row_size-1));
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaMalloc( (void **) &d_result, numBlocks*sizeof(double) );
	h_result = new double[numBlocks];
	
	timeval t;
	double t1,t2,t3,t4;
	
	gettimeofday(&t, NULL);
	t1 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	
	
	romberg<<< numBlocks, numThreadsPerBlock, row_size*numThreadsPerBlock*sizeof(double) >>>(A,B,row_size,d_result);
	cudaThreadSynchronize();
	
	gettimeofday(&t, NULL);
	t2 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	checkCUDAError("kernel invocation");
	cudaMemcpy( h_result, d_result, numBlocks*sizeof(double), cudaMemcpyDeviceToHost );
	checkCUDAError("memcpy");
	
	//for(int k = 0; k<(max_eval+1)*numBlocks; k++ )
	//	printf("%lf\t",h_result[k]);
	for(int k=0;k<numBlocks;k++)
		sum+=h_result[k];
	printf("TIME : %lf ms with ans = %lf\n\n\n",t2-t1,sum);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
