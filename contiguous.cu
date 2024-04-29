#include <cuda_runtime.h>
#include <iostream>
#include <ctime>

__global__ void contiguousPartition(int *adjacencyMatrix, int *weights, int *partitionAssignments, int numVertices, int numPartitions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertices) {
        int verticesPerPartition = numVertices / numPartitions; // Calculate number of vertices per partition

        int partitionCounter = 0; // Counter to keep track of partition index
        //int verticesAssigned = 0; // Counter to keep track of vertices assigned to partitions

        // Calculate partition index for current thread
        partitionCounter = tid / verticesPerPartition;
        //verticesAssigned = tid % verticesPerPartition;//

        // Assign current vertex to current partition
        partitionAssignments[tid] = partitionCounter;

        // Check edges between current vertex and all previous vertices
        for (int i = 0; i < tid; ++i) {
            if (adjacencyMatrix[tid * numVertices + i] && weights[tid * numVertices + i] == weights[i * numVertices + tid]) {
                partitionAssignments[i] = partitionAssignments[tid]; // Ensure vertices with same weight are in the same partition
            }
        }
    }
}

int main() {
    const int numVertices = 20;
    const int numPartitions = 4;

    int adjacencyMatrix[numVertices][numVertices] = {{0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0}};

    int weights[numVertices][numVertices] = {{0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0}};

    int partitionAssignments[numVertices];
    int *d_adjacencyMatrix, *d_weights, *d_partitionAssignments;

    cudaMalloc((void **)&d_adjacencyMatrix, numVertices * numVertices * sizeof(int));
    cudaMalloc((void **)&d_weights, numVertices * numVertices * sizeof(int));
    cudaMalloc((void **)&d_partitionAssignments, numVertices * sizeof(int));

    cudaMemcpy(d_adjacencyMatrix, adjacencyMatrix, numVertices * numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, numVertices * numVertices * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (numVertices + blockSize - 1) / blockSize;


    // Measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    contiguousPartition<<<numBlocks, blockSize>>>(d_adjacencyMatrix, d_weights, d_partitionAssignments, numVertices, numPartitions);

    // Measure time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(partitionAssignments, d_partitionAssignments, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Partition Assignments:" << std::endl;
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Vertex " << i << ": Partition " << partitionAssignments[i] << std::endl;
    }

    cudaFree(d_adjacencyMatrix);
    cudaFree(d_weights);
    cudaFree(d_partitionAssignments);

    return 0;
}
