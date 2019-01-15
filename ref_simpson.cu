#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define warp_size 32
#define Hwarp_size 16
#define N_points 33554432
#define A 0
#define B 15

void checkCUDAError(const char* msg);

__host__ __device__ inline double f(double x)
{
	return exp(x)*sin(x);
}

__global__ void fn_evalCalc(double *fn_eval, double a, double b)	//N_points/numBlocks should be a integer.
{
	extern __shared__ double local_array[];
	double step = (b-a)/N_points, mult, sum = 0.0, diff = (b-a)/gridDim.x;
	int eval = N_points/gridDim.x;
	//b = a + (blockIdx.x+1)*diff;
	a += blockIdx.x*diff;
	
	for(int k = threadIdx.x; k < eval; k += blockDim.x)
	{
		mult = (k%2==0)?2.0:4.0;
		sum += mult*f(a + step*k);
	}
	local_array[threadIdx.x] = sum;
	__syncthreads();
	
	//BlockReduce.
	for(int s = 1; s < blockDim.x; s *= 2) 
	{
      if ((threadIdx.x % (2*s)) == 0) 
            local_array[threadIdx.x] += local_array[threadIdx.x + s];
      __syncthreads();
    }
    
    if(!threadIdx.x)
		fn_eval[blockIdx.x] = local_array[threadIdx.x];
	
}



__global__ void globalReduce(double *fn_eval, double a, double b, int size)
{
		extern __shared__ double local_array[];
		double step = (b-a)/N_points;
		if(threadIdx.x < size)
			local_array[threadIdx.x] = fn_eval[threadIdx.x];

		
		for(int s = 1; s < blockDim.x; s *= 2) 
		{
			if ((threadIdx.x % (2*s)) == 0) 
				local_array[threadIdx.x] += local_array[threadIdx.x + s];
			__syncthreads();
		}
		if(!threadIdx.x)
			fn_eval[0] = step*(local_array[threadIdx.x] + f(b) - f(a))/3;
}


int main( int argc, char** argv)
{
	double sum=0.0,*d_fn_eval;
	int numBlocks = 128, numThreadsPerBlock = 64; //keep numBlocks within 1024
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	timeval t;
	double t1,t2;
	
	cudaMalloc( (void **) &d_fn_eval, sizeof(double) );
	gettimeofday(&t, NULL);
	t1 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	
	
	fn_evalCalc<<< numBlocks, numThreadsPerBlock, numThreadsPerBlock*sizeof(double) >>>(d_fn_eval,A,B);
	globalReduce<<< 1, numBlocks, numBlocks*sizeof(double) >>>(d_fn_eval,A,B,numBlocks);
	cudaThreadSynchronize();
	
	gettimeofday(&t, NULL);
	t2 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	checkCUDAError("kernel invocation");
	cudaMemcpy( &sum, d_fn_eval, sizeof(double), cudaMemcpyDeviceToHost );
	checkCUDAError("memcpy");

	//for(int k = 0; k<N_points; k++ )
	//	printf("%lf\t",h_fn_eval[k]);
	//for(int k=0;k<numBlocks;k++)
	//	sum+=h_debug_output[k];
	printf("%lf~~~TIME : %lf ms\n\n\n",sum,t2-t1);//,sum);
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
