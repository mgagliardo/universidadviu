/* Programa e1.c */
#include <omp.h>

int main() {
    #pragma omp parallel
    printf("Hola mundo\n");
    exit (0);
}
