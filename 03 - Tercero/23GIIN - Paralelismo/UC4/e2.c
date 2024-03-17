/* Programa e2.c */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int cantidad_hilos = 6;
    omp_set_num_threads(cantidad_hilos);
    #pragma omp parallel
    printf("Hola mundo\n");
    exit (0);
}
