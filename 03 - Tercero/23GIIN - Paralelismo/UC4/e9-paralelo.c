#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 5

int main() {
   /* DECLARACION DE VARIABLES */
   float A[N][N], B[N][N], C[N][N]; // declaracion de matrices de tamaño NxN
   int i, j, m; // indices para la multiplicacion de matrices

   /* LLENAR LAS MATRICES CON NUMEROS ALEATORIOS */
   srand(time(NULL));
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
         A[i][j] = (rand() % 10);
         B[i][j] = (rand() % 10);
      }
   }

   /* MULTIPLICACION DE LAS MATRICES */
   #pragma omp parallel for collapse(2) private(m) shared(A, B, C)
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
         C[i][j] = 0.0; // colocar el valor inicial para el componente C[i][j] = 0
         for (m = 0; m < N; m++) {
            C[i][j] += A[i][m] * B[m][j];
         }
      }
   }

   /* IMPRIMIR LA MATRIZ RESULTANTE */
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
         printf("C: %f ", C[i][j]);
      }
      printf("\n");
   }

   /* FINALIZAR EL PROGRAMA */
   return 0;
}
