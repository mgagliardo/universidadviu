#include <stdio.h>

#define N 100000000

int num[N];

float func_average() {
    int i;
    float sum = 0.0;
    for(i = 0; i < N; ++i) {
        sum += num[i];
    }
    return sum / N;
}

int main() {
    int i;
    float average;
    for (i=0;i<N;i++) {
         num[i]=i;
   	}
    average = func_average();
    printf("Average = %f\n", average);
    return 0;
}
