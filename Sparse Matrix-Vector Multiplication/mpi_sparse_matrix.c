#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double **spmatrix;  // Sparse matrix as a 2D array
double *x;          // Input vector
double *y;          // Output vector

void allocate_matrix(double ***matrix, int rows, int cols) {
    *matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        (*matrix)[i] = (double *)malloc(cols * sizeof(double));
    }
}

void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to initialize the result vector
void init_result(double *result, int rows) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
    }
}

// Function to initialize the vector
void init_vector(double *vector, int cols) {
    for (int i = 0; i < cols; i++) {
        vector[i] = (rand() % 10) + 1;  
    }
}

// Function to generate sparse matrix based on sparsity percentage
void generateSparseMatrix(double **matrix, int rows, int cols, double sparsity) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((double)rand() / RAND_MAX >= sparsity) {
                matrix[i][j] = (double)(1 + (rand() % 10)); 
            } else {
                matrix[i][j] = 0.0;
            }
        }
    }
}

// Sparse Matrix-Vector Multiplication
void sparseMatrixVectorMultiply(double **matrix, double *vec, double *res, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] != 0.0) {
                sum += matrix[i][j] * vec[j];
            }
        }
        res[i] = sum;
    }
}

// Calculate maximum value in the output vector
double calculateMax(double *vec, int size) {
    double max_val = vec[0];
    for (int i = 1; i < size; i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }
    return max_val;
}

// Function to calculate the sum of the vector y
double calculateVectorSum(double *vec, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vec[i];
    }
    return sum;
}

// Function to find the minimum value in the vector y
double findMinValue(double *vec, int size) {
    double min_val = vec[0];
    for (int i = 1; i < size; i++) {
        if (vec[i] < min_val) {
            min_val = vec[i];
        }
    }
    return min_val;
}

void calculateRowSums(double **matrix, int rows, int cols, double *row_sums) {
    for (int i = 0; i < rows; i++) {
        row_sums[i] = 0.0; // Initialize sum for the current row
        for (int j = 0; j < cols; j++) {
            row_sums[i] += matrix[i][j]; // Add the value to the row sum
        }
    }
}

void calculateColumnSums(double **matrix, int rows, int cols, double *column_sums) {
    for (int j = 0; j < cols; j++) {
        column_sums[j] = 0.0; // Initialize sum for the current column
        for (int i = 0; i < rows; i++) {
            column_sums[j] += matrix[i][j]; // Add the value to the column sum
        }
    }
}

double calculateMatrixSum(double **matrix, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += matrix[i][j];
        }
    }
    return sum;
}

double *calculateRowAverages(double *row_sums, int cols) {
    double *row_avgs = (double *)malloc(cols * sizeof(double));
    for (int i = 0; i < cols; i++) {
        row_avgs[i] = row_sums[i] / cols;
    }
    return row_avgs;
}

double *calculateColumnAverages(double *col_sums, int rows) {
    double *col_avgs = (double *)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        col_avgs[i] = col_sums[i] / rows;
    }
    return col_avgs;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int original_size, fixed_size = 20000;
    double sparsity;
    srand(time(NULL) + rank);

    // Master process handles input
    if (rank == 0) {
    	printf("size = %d\n",size);
    	printf("Rank 0: Entering input phase\n");
    	printf("Enter the number of rows or columns for square matrix: ");
    	fflush(stdout);  
    	scanf("%d", &original_size);
    	printf("Rank 0: Matrix size input received: %d\n", original_size);

    	printf("Enter the sparsity percentage (0-1): ");
    	fflush(stdout);
    	scanf("%lf", &sparsity);
    	printf("Rank 0: Sparsity input received: %f\n", sparsity);
    }

    // Broadcast inputs to all processes
    MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int num_loops = original_size / fixed_size;
    int rows_per_process = fixed_size / size;
	
    // Allocate memory for vectors and matrices
    allocate_matrix(&spmatrix, rows_per_process, fixed_size);
    x = (double *)malloc(fixed_size * sizeof(double));
    y = (double *)malloc(rows_per_process * sizeof(double));
    double global_max = 0.0, global_min = 0.0, global_sum = 0.0;
    double local_max, local_min, local_sum;
    
    double *global_row_sums = NULL;
    double *global_column_sums = (double *)malloc(fixed_size * sizeof(double));
    double local_matrix_sum = 0.0, global_matrix_sum = 0.0;

    if (rank == 0) {
        global_row_sums = (double *)malloc(fixed_size * sizeof(double));
    }

    double start_time, end_time;
    if (rank == 0) start_time = MPI_Wtime();

    double *local_row_sums = (double *)malloc(fixed_size * sizeof(double));
    double *local_column_sums = (double *)malloc(fixed_size * sizeof(double));
    for (int loop = 0; loop < num_loops; loop++) {
        if (rank == 0) {
            init_vector(x, fixed_size);
        }
        MPI_Bcast(x, fixed_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        generateSparseMatrix(spmatrix, rows_per_process, fixed_size, sparsity);
        init_result(y, rows_per_process);
  
	MPI_Barrier(MPI_COMM_WORLD);
        sparseMatrixVectorMultiply(spmatrix, x, y, rows_per_process, fixed_size);
        MPI_Barrier(MPI_COMM_WORLD);

        local_max = calculateMax(y, rows_per_process);
        local_min = findMinValue(y, rows_per_process);
        local_sum = calculateVectorSum(y, rows_per_process);

        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        calculateRowSums(spmatrix, rows_per_process, fixed_size, local_row_sums);
        calculateColumnSums(spmatrix, rows_per_process, fixed_size, local_column_sums);
        local_matrix_sum = calculateMatrixSum(spmatrix, rows_per_process, fixed_size);

        MPI_Reduce(local_row_sums, global_row_sums, fixed_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(local_column_sums, global_column_sums, fixed_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_matrix_sum, &global_matrix_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);

    }

    if (rank == 0) {
    	double *row_averages = calculateRowAverages(global_row_sums, fixed_size);
        double *column_averages = calculateColumnAverages(global_column_sums, fixed_size);
        end_time = MPI_Wtime();
        printf("Maximum value in the result vector: %f\n", global_max);
        printf("Minimum value in the result vector: %f\n", global_min);
        printf("Sum of the result vector: %f\n", global_sum);
        printf("Matrix sum: %f\n", global_matrix_sum);
        printf("Row averages calculated\n");
        printf("Column averages calculated\n");
        printf("Execution time: %f seconds\n", end_time - start_time);
        
        free(global_row_sums);
        free(row_averages);
        free(column_averages);

    }
    free(global_column_sums);
    free(x);
    free(y);
    free(local_row_sums);
    free(local_column_sums);
    free_matrix(spmatrix, rows_per_process);  
    MPI_Finalize();
    return 0;
}

