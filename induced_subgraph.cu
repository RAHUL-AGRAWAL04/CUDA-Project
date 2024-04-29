#include <stdio.h>

#define N 5 // Number of vertices

__global__ void generateSubgraphCSR(int* row_ptr, int* col_indices, int* values, int* sub_row_ptr, int* sub_col_indices, int* sub_values, int* vertices, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numVertices) {
        int vertex = vertices[idx];
        sub_row_ptr[idx] = sub_row_ptr[idx - 1];
        for (int j = row_ptr[vertex]; j < row_ptr[vertex + 1]; ++j) {
            int neighbor = col_indices[j];
            for (int k = 0; k < numVertices; ++k) {
                if (neighbor == vertices[k]) {
                    sub_col_indices[sub_row_ptr[idx]] = k;
                    sub_values[sub_row_ptr[idx]] = values[j];
                    sub_row_ptr[idx]++;
                    break;
                }
            }
        }
    }
}

int main() {
    int row_ptr[N+1] = {0, 2, 5, 8, 11, 14}; // Row pointers for the CSR format
    int col_indices[14] = {1, 3, 0, 2, 4, 0, 1, 3, 0, 2, 4, 1, 3, 4}; // Column indices for the CSR format
    int values[14] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // Values for the CSR format

    int sub_row_ptr[N];
    int sub_col_indices[N * N];
    int sub_values[N * N];

    int* d_row_ptr;
    int* d_col_indices;
    int* d_values;
    int* d_sub_row_ptr;
    int* d_sub_col_indices;
    int* d_sub_values;
    int* d_vertices;

    cudaMalloc((void**)&d_row_ptr, (N+1) * sizeof(int));
    cudaMalloc((void**)&d_col_indices, 14 * sizeof(int));
    cudaMalloc((void**)&d_values, 14 * sizeof(int));
    cudaMalloc((void**)&d_sub_row_ptr, N * sizeof(int));
    cudaMalloc((void**)&d_sub_col_indices, N * N * sizeof(int));
    cudaMalloc((void**)&d_sub_values, N * N * sizeof(int));
    cudaMalloc((void**)&d_vertices, N * sizeof(int));

    cudaMemcpy(d_row_ptr, row_ptr, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, 14 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, 14 * sizeof(int), cudaMemcpyHostToDevice);

    int vertices[] = {0, 1, 3}; // Example vertices for the induced subgraph

    cudaMemcpy(d_vertices, vertices, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + 255) / 256;
    int numThreadsPerBlock = 256;

    generateSubgraphCSR<<<numBlocks, numThreadsPerBlock>>>(d_row_ptr, d_col_indices, d_values, d_sub_row_ptr, d_sub_col_indices, d_sub_values, d_vertices, N);

    cudaMemcpy(sub_row_ptr, d_sub_row_ptr, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_col_indices, d_sub_col_indices, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sub_values, d_sub_values, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Induced Subgraph in CSR format:\n");
    printf("sub_row_ptr: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", sub_row_ptr[i]);
    }
    printf("\nsub_col_indices: ");
    for (int i = 0; i < N * N; ++i) {
        printf("%d ", sub_col_indices[i]);
    }
    printf("\nsub_values: ");
    for (int i = 0; i < N * N; ++i) {
        printf("%d ", sub_values[i]);
    }
    printf("\n");

    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_sub_row_ptr);
    cudaFree(d_sub_col_indices);
    cudaFree(d_sub_values);
    cudaFree(d_vertices);

    return 0;
}
