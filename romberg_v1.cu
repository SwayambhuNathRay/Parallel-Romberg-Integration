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

__global__ void romberg(double a, double b, int max_eval, double *result)
{
	extern __shared__ double local_array[];
	double diff = (b-a)/gridDim.x, step;
	b = a + (blockIdx.x+1)*diff;
	a += blockIdx.x*diff;
	
	step = (b-a)/max_eval;
	for(int k = threadIdx.x; k < max_eval+1; k += blockDim.x)
		local_array[k] = f(a + step*k);
	
	//for(int k = threadIdx.x; k < max_eval+1; k += blockDim.x)
	//	result[blockIdx.x*(max_eval+1)+k] = local_array[k];

	if(threadIdx.x < 13)
	{
		int inc = 1<<(12-threadIdx.x);
		double sum = 0.0;
		for(int k = 0;k <= max_eval;k = k+inc)
		{
			sum += 2.0*local_array[k];
		}
		sum -= (local_array[0] + local_array[max_eval]);
		sum *= (b-a)/(1<<(threadIdx.x+1));
		local_array[threadIdx.x] = sum;
	}
	
	if(!threadIdx.x)
	{
		double romberg_table[13];
		for(int k=0;k<13;k++)
			romberg_table[k] = local_array[k];
		
		for(int col = 0 ; col < 12 ; col++)
		{
			for(int row = 12; row > col; row--)
			{
				romberg_table[row] = romberg_table[row] + (romberg_table[row] - romberg_table[row-1])/((1<<(2*col+1))-1);
			}
		}
		result[blockIdx.x] = romberg_table[12];
	}

}


int main( int argc, char** argv)
{
	double *d_result, *h_result,sum=0.0;
	int numBlocks = 128, numThreadsPerBlock = 64, max_eval = 4096;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaMalloc( (void **) &d_result, numBlocks*sizeof(double) );
	h_result = new double[numBlocks];
	
	timeval t;
	double t1,t2;
	
	gettimeofday(&t, NULL);
	t1 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	
	
	romberg<<< numBlocks, numThreadsPerBlock, (max_eval+1)*sizeof(double) >>>(A,B,max_eval,d_result);
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
