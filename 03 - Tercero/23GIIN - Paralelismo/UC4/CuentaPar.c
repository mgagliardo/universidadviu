//------------------------------------------------------------------+
// CuentaPar.c: Cuenta aparaciones de un numero en un arreglo muy   |
//                  grande.Version paralela simple                  |
//                      ESQUELETO                                   |
//------------------------------------------------------------------+

#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"


#define MAX_ENTERO 1000
#define NUM_VECTORES 10000 // Simula vector todavia mayor
#define NUM_BUSCADO 8

//--------------------------------------------------------------------
void esclavo(int yo, int NumProcesos, int Cardinalidad) {
    int offset = Cardinalidad / (NumProcesos-1);
    int *buff;
    int i = 0;
    int aux = 0;
    int repetidos[1] = {0};
    MPI_Status estado;
	
    assert((buff =(int *)malloc(sizeof(int)*offset))!=NULL);
    MPI_Recv(buff, offset, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);

    // se repite NUM_VECTORES veces
    for(aux =0; aux < NUM_VECTORES; aux++){
        for (i=0; i<offset; i++){
            if(buff[i] == NUM_BUSCADO)
                repetidos[0]++;
	    }
    }
    MPI_Send(repetidos, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
}
//--------------------------------------------------------------------

void maestro (int NumProcesos, int Cardinalidad) {
    int i, j;
    int totalNumVeces = 0;
    int *vector;
    // int vector[10] = {9,8,7,6,5,4,3,2,1,0};
    int result[1] = {0};
    int offset = Cardinalidad / (NumProcesos-1);
    MPI_Status estado;
    struct timeval t0, tf, t;
    
    // Inicializar el vector
    assert((vector =(int *)malloc(sizeof(int)*Cardinalidad))!=NULL);
    
    for (i=0; i<Cardinalidad; i++)
        vector[i] = random() % MAX_ENTERO;

    assert (gettimeofday (&t0, NULL) == 0);
    
    j = 0;
    for(i=1; i<NumProcesos; i++){
	// Repartir trabajo
	//int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
        MPI_Send(vector + (j*offset), offset, MPI_INT, i, 1, MPI_COMM_WORLD);
        // Computar mi trozo
        // Recoger resultados
	j++;
    }

    for(i=1; i<NumProcesos; i++){
	MPI_Recv(result, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &estado);
	totalNumVeces += result[0];
    }

    assert (gettimeofday (&tf, NULL) == 0);
    timersub(&tf, &t0, &t);
    printf ("Numero de veces que aparece el %d = %d\n",
        NUM_BUSCADO, totalNumVeces);
    printf ("tiempo total = %ld:%3ld\n", t.tv_sec, t.tv_usec/1000);
}

//--------------------------------------------------------------------
int main(int argc, char * argv[]) {
    int yo, numProcesos, laCardinalidad;

    if (argc != 2) {
      printf("Uso: cuentaPar cardinalidad \n");
      return 0;
    }
    laCardinalidad = atoi(argv[1]);
    assert(laCardinalidad > 0);
    setbuf(stdout, NULL);
    MPI_Init( & argc, & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, & yo);
	MPI_Comm_size (MPI_COMM_WORLD, & numProcesos);
    if (yo == 0) {
      maestro(numProcesos, laCardinalidad);
    } else {
      esclavo(yo, numProcesos, laCardinalidad);
    }
    MPI_Finalize();
    return 0;
}
