#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
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


__global__ void rombergOMP(double a, double b, int row_size, double *omp_f)
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
		
		//local_array[threadIdx.x] = sum;
		omp_f[blockIdx.x*row_size + threadIdx.x] = sum;
	}
	
	
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
	double *h_omp_f,*d_omp_f,*d_result, *h_result,sum=0.0,ompA,ompB = B,gpuA = A,gpuB;
	int numBlocks = 128,numBlocksOMP, numThreadsPerBlock = 64, row_size = 13, max_eval = (1<<(row_size-1)), core = 6;
	double my_sum[6] = {0.0};
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaStream_t streams[2];
	omp_set_num_threads(core);
	numBlocksOMP = numBlocks/4;		//TODO : hyperparameter
	gpuB = B/4;						//TODO : hyperparameter
	ompA = gpuB;
    cudaMalloc( (void **) &d_result, numBlocks*sizeof(double) );
    cudaHostAlloc( (void**)&d_omp_f, numBlocksOMP*row_size*sizeof(double), cudaHostAllocDefault );
    //cudaMalloc( (void **) &d_omp_f, numBlocksOMP*row_size*sizeof(double) );
	h_result = new double[numBlocks];
	h_omp_f = new double[numBlocksOMP*row_size];
	timeval t;
	double t1,t2;
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);
	gettimeofday(&t, NULL);
	t1 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	
	rombergOMP<<< numBlocksOMP, numThreadsPerBlock, row_size*numThreadsPerBlock*sizeof(double),streams[0] >>>(ompA,ompB,row_size,d_omp_f);
	cudaStreamSynchronize (streams[0]);
	//cudaThreadSynchronize();
	
	cudaMemcpyAsync( h_omp_f, d_omp_f, numBlocksOMP*row_size*sizeof(double), cudaMemcpyDeviceToHost, streams[0] );
	romberg<<< numBlocks-numBlocksOMP, numThreadsPerBlock, row_size*numThreadsPerBlock*sizeof(double), streams[1] >>>(gpuA,gpuB,row_size,d_result);

	#pragma omp parallel for default(shared) schedule(static)
	for(int p = 0; p < numBlocksOMP; p++)
	{
		//double t3; timeval _t;
		//gettimeofday(&_t, NULL);
		//t3 = _t.tv_sec*1000.0 + (_t.tv_usec/1000.0);
		double romberg_table[row_size];
		romberg_table[0] = h_omp_f[p*row_size];
		for(int k = 1; k < row_size; k++)
			romberg_table[k] = romberg_table[k-1] + h_omp_f[p*row_size + k];
		for(int k = 0; k < row_size; k++)	
			romberg_table[k] *= ((ompB-ompA)/numBlocksOMP)/(1<<(k+1));
			
		for(int col = 0 ; col < row_size-1 ; col++)
		{
			for(int row = row_size-1; row > col; row--)
			{
				romberg_table[row] = romberg_table[row] + (romberg_table[row] - romberg_table[row-1])/((1<<(2*col+1))-1);
			}
		}
		
		my_sum[omp_get_thread_num()] += romberg_table[row_size-1];
		//gettimeofday(&_t, NULL);
		//t3 = _t.tv_sec*1000.0 + (_t.tv_usec/1000.0) - t3;
		//printf("p = %d, time = %lf ms\n",p,t3);
	}
	
	
	
	
	cudaThreadSynchronize();
	gettimeofday(&t, NULL);
	
	t2 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
	checkCUDAError("kernel invocation");
	cudaMemcpy( h_result, d_result, numBlocks*sizeof(double), cudaMemcpyDeviceToHost );
	checkCUDAError("memcpy");
	
	//for(int k = 0; k<(max_eval+1)*numBlocks; k++ )
	//	printf("%lf\t",h_result[k]);
	for(int k=0;k<numBlocksOMP;k++)
		sum+=my_sum[k];
	for(int k=0;k<numBlocks-numBlocksOMP;k++)
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
