#include <thrust/sequence.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <sys/time.h>


#define pi_f  3.14159265358979f                 // Greek pi in single precision

struct sin_functor
{
    __host__ __device__
    float operator()(float x) const
    {
        return x*sin(x);
    }
};

int main(void)
{
    int M = 12;                          // --- Maximum number of Romberg iterations

    float a     = 0.f;                  // --- Lower integration limit
    float b     = 1000.f;                  // --- Upper integration limit

    float hmin  = (b-a)/pow(2.f,M-1);   // --- Minimum integration step size 

    // --- Define the matrix for Romberg approximations and initialize to 1.f 
    
    timeval t;
	double t1,t2,t3,t4;
	
	gettimeofday(&t, NULL);
	t1 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);
    thrust::host_vector<float> R(M*M,1.f);

    for (int k=0; k<M; k++) {

        float h = pow(2.f,k-1)*hmin;    // --- Step size for the k-th row of the Romberg matrix

        // --- Define integration nodes
        int N = (int)((b - a)/h) + 1;
        thrust::device_vector<float> d_x(N);
        thrust::sequence(d_x.begin(), d_x.end(), a, h);

        // --- Calculate function values
        thrust::device_vector<float> d_y(N);
        thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), sin_functor());

        // --- Calculate integral
        R[k*M] = (.5f*h) * (d_y[0] + 2.f*thrust::reduce(d_y.begin() + 1, d_y.begin() + N - 1, 0.0f) + d_y[N-1]);

    }

    // --- Compute the k-th column of the Romberg matrix
    for (int k=1; k<M; k++) { 

        // --- The matrix of Romberg approximations is triangular!
        for (int kk=0; kk<(M-k+1); kk++) { 

            // --- See the Romberg integration algorithm
            R[kk*M+k] = R[kk*M+k-1] + (R[kk*M+k-1] - R[(kk+1)*M+k-1])/(pow(4.f,k)-1.f); 

        } 

    }
    
    gettimeofday(&t, NULL);
	t2 = t.tv_sec*1000.0 + (t.tv_usec/1000.0);

    // --- Define the vector Rnum for numerical approximations
    thrust::host_vector<float> Rnum(M); 
    thrust::copy(R.begin(), R.begin() + M, Rnum.begin());
    
   
	printf("TIME : %lf ms\n",t2-t1);

    for (int i=0; i<M; i++) printf("%i %f\n",i,Rnum[i]);

    //getchar();

    return 0;
}
