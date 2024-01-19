#include "mpi.h"
#include <math.h>
#include <stdio.h>

#define STEPS 5

double f(double a) {
  return (4.0 / (1.0 + a * a));
}

int main(int argc, char * argv[]) {
  int n, myid, numprocs, i, j;
  double PI25DT = 3.141592653589793238462643;
  double mypi, pi, h, sum, x;
  double startwtime = 0.0, endwtime;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Init( & argc, & argv);
  MPI_Comm_size(MPI_COMM_WORLD, & numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, & myid);
  MPI_Get_processor_name(processor_name, & namelen);
  fprintf(stderr, "Process %d running on %s\n", myid, processor_name);
  if (myid == 0) {
    n = 1000000;
    startwtime = MPI_Wtime();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (j = 0; j < STEPS; j++) {
    MPI_Bcast( & n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = myid + 1; i <= n; i += numprocs) {
      x = h * ((double) i - 0.5);
      sum += f(x);
    }
    mypi = h * sum;
    MPI_Reduce( & mypi, & pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (myid == 0) {
    endwtime = MPI_Wtime();
    printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
    printf("wall clock time = %f\n", endwtime - startwtime);
  }
  MPI_Finalize();
  return (0);
}
