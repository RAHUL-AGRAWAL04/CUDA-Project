// %%cuda
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <cuda.h>

__global__ void computeSubgraph(int n, int* rowPtr, int* colIdx, bool* isSelected, int* subRowPtr, int* subColIdx, int* nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n && isSelected[tid]) {
        int start = rowPtr[tid];
        int end = rowPtr[tid + 1];
        //subRowPtr[0] = 0;

        int count = 0;

        for (int j = start; j < end; j++) {
            int col = colIdx[j];
            if (isSelected[col]) {
                int index = atomicAdd(nnz, 1);
                subColIdx[index] = col;
                count++;
            }
        }

        __syncthreads();
        subRowPtr[0] = 0;
        for (int i=0;i<n;i++){
            if(tid == i){
                subRowPtr[tid+1] = subRowPtr[tid]+count; //rectify
                __syncthreads();
            }
            
        }

        //subRowPtr[tid + 1] = *nnz;
        //subRowPtr[tid + 1] = subRowPtr[tid] +count;
    }
}

void getSubgraph(int n, std::vector<int>& rowPtr, std::vector<int>& colIdx, std::vector<int>& vertices) {
    // Copy data to device
    thrust::device_vector<int> d_rowPtr(rowPtr.begin(), rowPtr.end());
    thrust::device_vector<int> d_colIdx(colIdx.begin(), colIdx.end());
    thrust::device_vector<int> d_vertices(vertices.begin(), vertices.end());

    // Initialize device vectors
    thrust::device_vector<int> d_subRowPtr(n + 1, 0);
    thrust::device_vector<int> d_subColIdx(colIdx.size(), 0);
    thrust::device_vector<bool> d_isSelected(n, false);
    for (int v : vertices) {
        d_isSelected[v] = true;
    }

    int nnz = 0;
    int* d_nnz;
    cudaMalloc((void**)&d_nnz, sizeof(int));
    cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Kernel call
    computeSubgraph<<<numBlocks, blockSize>>>(n, thrust::raw_pointer_cast(&d_rowPtr[0]), thrust::raw_pointer_cast(&d_colIdx[0]), 
                                               thrust::raw_pointer_cast(&d_isSelected[0]), thrust::raw_pointer_cast(&d_subRowPtr[0]), 
                                               thrust::raw_pointer_cast(&d_subColIdx[0]), d_nnz);
    cudaDeviceSynchronize();

    cudaMemcpy(&nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost);

    // Printing the subgraph
    std::cout << "Induced Subgraph:\n";
    std::cout << "rowPtr: ";
    thrust::copy(d_subRowPtr.begin(), d_subRowPtr.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "colIdx: ";
    thrust::copy(d_subColIdx.begin(), d_subColIdx.begin() + nnz, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    cudaFree(d_nnz);
}

int main() {
    //std::vector<int> rowPtr = {0, 2, 5, 8, 10};
    //std::vector<int> colIdx = {1, 3, 0, 2, 4, 1, 3, 2, 4, 3};
    //std::vector<int> vertices = {0, 2, 4};


    std::vector<int> rowPtr = {0,3,4,5};
    std::vector<int> colIdx = {1, 2,3,2,3};
    std::vector<int> vertices = {0, 1,2};

    getSubgraph(3, rowPtr, colIdx, vertices);

    return 0;
}
