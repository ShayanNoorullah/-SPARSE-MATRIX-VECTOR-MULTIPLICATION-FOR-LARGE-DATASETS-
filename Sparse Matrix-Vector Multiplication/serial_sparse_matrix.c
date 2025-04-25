#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// Sparse Matrix Representation using a 2D array
double **spmatrix;  // Sparse matrix as a 2D array
double *x;          // Input vector
double *y;          // Output vector

// Function to initialize the result vector
void init_result(double *result, int rows) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;  
    }
}

// Function to initialize the vector
void init_vector(double *vector, int cols) {
    for (int i = 0; i < cols; i++) {
        vector[i] = (rand() % 10)+ 1; 
    }
}

// Function to generate sparse matrix based on sparsity percentage
void generateSparseMatrix(int rows, int cols, double sparsity) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < cols; j++) {
            if ((double)rand() / RAND_MAX >= sparsity) {
                spmatrix[i][j] = (double)(1 + (rand() % 10));
            } else {
                spmatrix[i][j] = 0.0;
            }
        }
    }
}

// Sparse Matrix-Vector Multiplication
void sparseMatrixVectorMultiply(int rows) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < rows; j++) { 
            if (spmatrix[i][j] != 0) {  
                sum += spmatrix[i][j] * x[j];
            }
        }
        y[i] = sum;
    }
}

// Calculate maximum value in the output vector
double calculateMax(int rows) {
    double max_val = y[0];
    for (int i = 1; i < rows; i++) {
        if (y[i] > max_val) {
            max_val = y[i];
        }
    }
    return max_val;
}

// Function to calculate the sum of the vector y
double calculateVectorSum(int rows) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        sum += y[i];
    }
    return sum;
}

// Function to find the minimum value in the vector y
double findMinValue(int rows) {
    double min_val = y[0];
    for (int i = 1; i < rows; i++) {
        if (y[i] < min_val) {
            min_val = y[i];
        }
    }
    return min_val;
}

// Function to calculate the row sums
double *calculateRowSums(int rows, int cols) {
    double *row_sums = (double *)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        row_sums[i] = 0.0;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            row_sums[i] += spmatrix[i][j];
        }
    }

    return row_sums;
}

// Function to calculate the column sums
double *calculateColumnSums(int rows, int cols) {
    double *col_sums = (double *)malloc(cols * sizeof(double));
    for (int i = 0; i < cols; i++) {
        col_sums[i] = 0.0;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            col_sums[j] += spmatrix[i][j];
        }
    }

    return col_sums;
}

// Function to calculate the matrix sum
double calculateMatrixSum(int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += spmatrix[i][j];
        }
    }
    return sum;
}

// Function to calculate row averages
double *calculateRowAverages(int rows, int cols) {
    double *row_avgs = (double *)malloc(rows * sizeof(double));
    double *row_sums = calculateRowSums(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        row_avgs[i] = row_sums[i] / cols; 
    }

    free(row_sums);
    return row_avgs;
}

// Function to calculate column averages
double *calculateColumnAverages(int rows, int cols) {
    double *col_avgs = (double *)malloc(cols * sizeof(double));
    double *col_sums = calculateColumnSums(rows, cols);
    
    for (int i = 0; i < cols; i++) {
        col_avgs[i] = col_sums[i] / rows;
    }

    free(col_sums);
    return col_avgs;
}

int main() {
    int rows = 20000, cols = 20000;
    double sparsity;
    int original_size;
    int fixed_size = 20000;
    srand(time(NULL));
    
    // User Input
    printf("Enter the number of rows or columns for square matrix: ");
    scanf("%d", &original_size);
    printf("Enter the sparsity percentage (0-1): ");
    scanf("%lf", &sparsity);

    // Allocate memory for vectors
    x = (double *)malloc(fixed_size * sizeof(double));
    y = (double *)malloc(fixed_size * sizeof(double));
    
    spmatrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        spmatrix[i] = (double *)malloc(cols * sizeof(double));
    }
	
    int num_loops = original_size / fixed_size;
    
    // Initializing variables
    double max_value = 0.0;
    double vector_sum = 0.0;
    double min_value = 0.0;
    double matrix_sum = 0.0;
    double *row_averages = NULL;
    double *col_averages = NULL;
 
    double start_time = clock();
    
    for(int i = 0 ; i < num_loops; i++) {
	init_vector(x, cols);
	generateSparseMatrix(rows, cols, sparsity);
	init_result(y, rows);
	    
	sparseMatrixVectorMultiply(rows);

	// Calculate Eval results
	int max = calculateMax(rows);
	if (max > max_value)
                    max_value = max;
	vector_sum += calculateVectorSum(rows);
	int min = findMinValue(rows);
	if (min < min_value)
                    min_value = min;
	matrix_sum += calculateMatrixSum(rows, cols);
	row_averages = calculateRowAverages(rows, cols);
	col_averages = calculateColumnAverages(rows, cols);
    }
    double end_time = clock();
    
    // Print results
    printf("Maximum value in the result vector: %f\n", max_value);
    printf("Sum of the result vector: %f\n", vector_sum);
    printf("Minimum value in the result vector: %f\n", min_value);
    printf("Matrix sum: %f\n", matrix_sum);
    printf("Row averages calculated\n");
    printf("Column averages calculated\n");
    printf("Execution time: %f seconds\n", (end_time - start_time) / CLOCKS_PER_SEC);

    // Free allocated memory
    free(x);
    free(y);
    free(row_averages);
    free(col_averages);
    for (int i = 0; i < rows; i++) {
        free(spmatrix[i]);
    }
    free(spmatrix);

    return 0;
}
