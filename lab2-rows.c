#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void init_mpi(int argc, char *argv[], int *comm_sz, int *my_rank) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, my_rank);
}

void initialize_vector(int *vec, int n, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            vec[i] = rand() % 100;
        }
    }
    MPI_Bcast(vec, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void input_matrix(int *matrix, int *local_mat, int rows, int cols, int local_mat_size, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = rand() % 100;
        }
    }
    MPI_Scatter(matrix, local_mat_size, MPI_INT, local_mat, local_mat_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void multiply_mat_vec(int *local_mat, int *local_vec, int *local_res, int local_rows, int cols) {
    for (int i = 0; i < local_rows; i++) {
        local_res[i] = 0;
        for (int j = 0; j < cols; j++) {
            local_res[i] += local_mat[i * cols + j] * local_vec[j];
        }
    }
}

int main(int argc, char *argv[]) {
    int comm_sz, my_rank;
    int rows, cols, vec_size;

    initialize_mpi(argc, argv, &comm_sz, &my_rank);

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    vec_size = cols;

    int local_rows = rows / comm_sz;
    int local_mat_size = local_rows * cols;

    int *vec = malloc(vec_size * sizeof(int));
    int *mat = NULL;
    int *result = NULL;
    int *local_res_vec = malloc(local_rows * sizeof(int));
    int *local_mat = malloc(local_mat_size * sizeof(int));

    if (my_rank == 0) {
        result = malloc(rows * sizeof(int));
        mat = malloc(cols * rows * sizeof(int));
    }

    input_vector(vec, vec_size, my_rank);
    input_matrix(mat, local_mat, rows, cols, local_mat_size, my_rank);

    double start_time = MPI_Wtime();
    multiply_mat_vec(local_mat, vec, local_res_vec, local_rows, cols);
    double end_time = MPI_Wtime();
    MPI_Gather(local_res_vec, local_rows, MPI_INT, result, local_rows, MPI_INT, 0, MPI_COMM_WORLD);

    double duration = end_time - start_time;
    double max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("time taken: %lf\n", max_duration * 1e3);
    }

    free(mat);
    free(result);
    free(local_res_vec);
    free(local_mat);
    free(vec);

    MPI_Finalize();
    return 0;
}
