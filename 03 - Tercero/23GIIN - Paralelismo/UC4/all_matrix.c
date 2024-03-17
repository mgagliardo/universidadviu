#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// Crea la matriz cuadrada N x N de numeros reales aleatorios
float *create_rand_matrix(int rows, int cols) {
  float *matrix = (float *)malloc(sizeof(float) * rows * cols);
  assert(matrix != NULL);

  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      matrix[i * cols + j] = (rand() / (float)RAND_MAX);
    }
  }
  return matrix;
}

// Realiza el computo de minimo, maximo y promedio (avg)
void compute_stats(float *row, int num_elements, float *min, float *max, float *avg) {
  *min = row[0];
  *max = row[0];
  *avg = 0.0;

  int i;
  for (i = 0; i < num_elements; i++) {
    *avg += row[i];
    if (row[i] < *min) {
      *min = row[i];
    }
    if (row[i] > *max) {
      *max = row[i];
    }
  }

  *avg /= num_elements;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (argc != 2) {
    fprintf(stderr, "Uso: %s <número_de_procesos>\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int num_processes = atoi(argv[1]);
  int rows_per_process = num_processes;
  int cols = num_processes;

  float *matrix = NULL;
  float *sub_row = (float *)malloc(sizeof(float) * cols);
  float min, max, avg;

  if (myrank == 0) {
    srand(time(NULL));
    matrix = create_rand_matrix(num_processes, num_processes);
    // Imprime la matriz generada
    printf("Matriz generada:\n");
    for (int i = 0; i < num_processes; i++) {
      for (int j = 0; j < num_processes; j++) {
        printf("%f ", matrix[i * num_processes + j]);
      }
      printf("\n");
    }
  }

  // Scatter filas de la matriz a cada proceso
  MPI_Scatter(matrix, cols, MPI_FLOAT, sub_row, cols, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Calcula las estadísticas para la fila asignada a cada proceso
  compute_stats(sub_row, cols, &min, &max, &avg);

  // Imprime las estadísticas para cada proceso
  printf("Process %d with row %d - min: %f; max: %f; avg: %f\n", myrank, myrank, min, max, avg);

  // Gather las estadísticas parciales para cada proceso en el proceso raíz
  float *stats_array = (float *)malloc(sizeof(float) * num_processes * 3); // 3 elementos por proceso (min, max, avg)
  assert(stats_array != NULL);
  MPI_Gather(&min, 1, MPI_FLOAT, stats_array, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gather(&max, 1, MPI_FLOAT, stats_array + num_processes, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gather(&avg, 1, MPI_FLOAT, stats_array + 2 * num_processes, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Proceso raíz calcula el mínimo, máximo y promedio global
  if (myrank == 0) {
    float global_min, global_max, global_avg;
    compute_stats(stats_array, num_processes, &global_min, &global_max, &global_avg);
    printf("Global minimum: %f; Global maximum: %f; Global average: %f\n", global_min, global_max, global_avg);
  }

  // Libera memoria
  if (myrank == 0) {
    free(matrix);
  }
  free(sub_row);
  free(stats_array);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
