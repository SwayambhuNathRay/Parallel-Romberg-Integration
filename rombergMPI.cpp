#include <iostream>
#include <cmath> 
#include <iomanip>
#include "mpi.h"
#include <cstdlib>
#include <sys/time.h>
#define A 0
#define B 15
#define row_size 13

using namespace std;

inline double f(double x)
{
	return exp(x)*sin(x);
}

inline unsigned int getFirstSetBitPos(int n)
{
   return log2((float)(n&-n))+1;
}

int main(int argc, char *argv[])
{
	int rank, size, max_eval = (1<<(row_size-1));
	double _time,_time_ref, step, diff, a, b, romberg_table[row_size],ans;
	MPI_Comm comm = MPI_COMM_WORLD;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	for(int k = 0; k < row_size; k++)
		romberg_table[k] = 0.0;
	if(rank == 0)
		_time = MPI_Wtime();
		
	diff = (B-A)/(size*1.0);
	b = A + (rank+1)*diff;
	a = A + rank*diff;
	
	step = (b-a)/(max_eval*1.0);
	
	romberg_table[0] = f(a) + f(b);
	for(int k = 1; k < max_eval; k++)
	{
		romberg_table[row_size - getFirstSetBitPos(k)] += 2.0*f(a + step*k);
		//cout<<"[k = "<<k<<"] Added f("<<a + step*k<<") = "<< 2.0*f(a + step*k) <<" and got romberg_table["<<row_size - getFirstSetBitPos(k)<<"] = "<<romberg_table[row_size - getFirstSetBitPos(k)]<<endl;
	}
	
	//if(rank==0)
	//{
	//	cout<<"step : "<<step<<", max_eval : "<<max_eval<<", a : "<<a<<", b : "<<b<<endl;
	//	for(int k = 0; k < row_size; k++)
	//		cout<<romberg_table[k]<<" ";// += romberg_table[k-1];
	//}
	
	for(int k = 1; k < row_size; k++)
		romberg_table[k] += romberg_table[k-1];
	for(int k = 0; k < row_size; k++)	
		romberg_table[k] *= (b-a)/(1<<(k+1));
		
	for(int col = 0 ; col < row_size-1 ; col++)
	{
		for(int row = row_size-1; row > col; row--)
		{
			romberg_table[row] = romberg_table[row] + (romberg_table[row] - romberg_table[row-1])/((1<<(2*col+1))-1);
		}
	}
	
	MPI_Reduce( &romberg_table[row_size-1], &ans, 1, MPI_DOUBLE, MPI_SUM, 0, comm );
	
	if(rank==0)
	{
		_time = MPI_Wtime() - _time;
		printf("\nTIME : %lf ms with ans = %lf\n\n\n",_time*1000.0,ans);
	}
	
	MPI_Finalize();
	return 0;
}
