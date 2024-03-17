//------------------------------------------------------------------+
// CuentaSec.c: Cuenta el numero de veces que aparece un numero en  |
//                      un vector muy grande.                       |
//------------------------------------------------------------------+
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define MAX_ENTERO      1000
#define NUM_VECTORES    10000   // Simula vector todavia mayor sin ocupar espacio
                                // de memoria
#define NUM_BUSCADO     8

int main(int argc, char * argv[]) {
  struct timeval t0, tf, t;
  int i, j, laCardinalidad, numVeces;
  int * vector;
  if (argc != 2) {
    printf("Uso: cuentaSec cardinalidad \n");
    return 0;
  }
  laCardinalidad = atoi(argv[1]);
  assert(laCardinalidad > 0);
  assert((vector = (int * ) malloc(sizeof(int) * laCardinalidad)) != NULL);
  for (i = 0; i < laCardinalidad; i++) {
    vector[i] = random() % MAX_ENTERO;
  }
  assert(gettimeofday( & t0, NULL) == 0);
  numVeces = 0;
  for (i = 0; i < NUM_VECTORES; i++)
    for (j = 0; j < laCardinalidad; j++)
      if (vector[j] == NUM_BUSCADO) numVeces++;
  assert(gettimeofday( & tf, NULL) == 0);
  timersub( & tf, & t0, & t);
  printf("Veces que aparece el %d = %d\n", NUM_BUSCADO, numVeces);
  printf("Tiempo de proceso: %ld:%3ld(seg:mseg)\n",
    t.tv_sec, t.tv_usec / 1000);
  return 0;
}
