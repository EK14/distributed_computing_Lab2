#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Init a vector with random values
void initialize_vector(int *vec, int n, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            vec[i] = rand() % 100;
        }
    }
}

// Multiply the matrix by the vector column-wise
void multiply_matrix_vector_cols(int *local_mat, int *local_vec, int *local_res, int rows, int cols, int start_col, int end_col) {
    for (int i = 0; i < rows; i++) {
        for (int j = start_col; j < end_col; j++) {
            local_res[i] += local_mat[i * cols + j] + local_vec[j];
        }
    }
}

// Initialize MPI and get the rank and size of the communicator
void initialize_mpi(int argc, char *argv[], int *comm_sz, int *my_rank) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, my_rank);
}

// Allocate and initialize memory for vectors and matrices
void allocate_memory(int **vec, int **mat, int **result, int **local_res_vec, int vec_size, int rows, int cols, int res_size, int my_rank) {
    *vec = malloc(vec_size * sizeof(int));
    *mat = malloc(rows * cols * sizeof(int));
    *local_res_vec = malloc(res_size * sizeof(int));

    for (int i = 0; i < res_size; i++) {
        (*local_res_vec)[i] = 0;
    }

    if (my_rank == 0) {
        *result = malloc(res_size * sizeof(int));
        for (int i = 0; i < res_size; i++) {
            (*result)[i] = 0;
        }
    }
}

// Broadcast the vector and matrix to all processes
void broadcast_data(int *vec, int *mat, int vec_size, int rows, int cols) {
    MPI_Bcast(vec, vec_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
}

// Calculate the column range for each process
void calculate_column_range(int my_rank, int comm_sz, int cols, int *start_col, int *end_col) {
    int local_cols = 1;
    if (my_rank < cols) {
        if (comm_sz <= cols) {
            local_cols = cols / comm_sz;
        }
        *start_col = my_rank * local_cols;
        *end_col = *start_col + local_cols;
        if (my_rank == comm_sz - 1) {
            *end_col = cols;
        }
    }
}

// Perform the matrix-vector multiplication and measure the time taken
void perform_multiplication_and_measure_time(int *mat, int *vec, int *local_res_vec, int rows, int cols, int start_col, int end_col, int my_rank, int res_size, int comm_sz) {
    double start_time = MPI_Wtime();
    if (my_rank < cols) {
        multiply_matrix_vector_cols(mat, vec, local_res_vec, rows, cols, start_col, end_col);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(local_res_vec, vec, res_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double duration = end_time - start_time;
    double max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("time taken: %lf\n", max_duration * 1e3);
    }
}

// Free allocated memory
void free_memory(int *mat, int *result, int *local_res_vec, int *vec) {
    free(mat);
    free(result);
    free(local_res_vec);
    free(vec);
}

int main(int argc, char *argv[]) {
    int comm_sz, my_rank;

    initialize_mpi(argc, argv, &comm_sz, &my_rank);

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int vec_size = cols;
    int res_size = rows;

    int *vec, *mat, *result, *local_res_vec;
    allocate_memory(&vec, &mat, &result, &local_res_vec, vec_size, rows, cols, res_size, my_rank);

    input_vector(vec, vec_size, my_rank);
    input_vector(mat, rows * cols, my_rank);
    broadcast_data(vec, mat, vec_size, rows, cols);

    int start_col, end_col;
    calculate_column_range(my_rank, comm_sz, cols, &start_col, &end_col);

    perform_multiplication_and_measure_time(mat, vec, local_res_vec, rows, cols, start_col, end_col, my_rank, res_size, comm_sz);

    free_memory(mat, result, local_res_vec, vec);

    MPI_Finalize();
    return 0;
}
