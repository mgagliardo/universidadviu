/* Programa e3.c */
#include <stdio.h>
#include <stdlib.h>

int main() {
    int comp=0;
    #pragma omp parallel {
        int priv=0;
        priv++;
        comp++;
        printf("Hola mundo priv %d comp %d\n",priv,comp);
    }
    exit (0);
}
