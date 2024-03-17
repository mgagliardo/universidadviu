#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

int i;
int nodo,numnodos;
int tam=32;

int main(int argc,char **argv)
{ MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&nodo);
  MPI_Comm_size(MPI_COMM_WORLD,&numnodos);

  MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel
  printf(" Soy el hilo %d de %d hilos dentro del procesador %d de %d procesadores\n",omp_get_thread_num(),omp_get_num_threads(),nodo,numnodos);

  MPI_Finalize();
}
