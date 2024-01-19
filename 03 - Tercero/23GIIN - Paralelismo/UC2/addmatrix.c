#include <stdio.h>

#define N 2000

//int a[N][N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 25, 33, 43, 53, 63}; 
//int b[N][N] = {10, 100, 25, 33, 43, 53, 63, 1, 2, 3, 4, 5, 6, 7, 8, 9}; 
//int sum[N][N];
int a[N][N], b[N][N], sum[N][N];

void func_sum() {
    int i, j;
    // Adding Two matrices
    for(i=0;i<N;++i) {
        for(j=0;j<N;++j) {
            sum[i][j]=a[i][j]+b[i][j];
        }
    }
}

void inicial() {
    int i,j;
    for (i=0;i<N;i++) {
        for (j=0;j<N;j++) {
            a[i][j]=i;
            b[i][j]=i+j;
        }
    } 
}

int main(){
    int i, j;
    inicial();
    func_sum();

    // Displaying the result
    printf("\nSum of two matrices: \n");
    for(i=0;i<N;++i) {
        for(j=0;j<N;++j) {
            printf("%d   ",sum[i][j]);
            if(j==N-1) {
                printf("\n\n");
            }
        }
    }
    return 0;
}
