// %%cuda
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <curand_kernel.h>

using namespace std;


struct Edge {
    int u, v, weight;
    Edge(int _u, int _v, int _weight) : u(_u), v(_v), weight(_weight) {}
};

class Graph {
public:
    int V;
    vector<vector<pair<int, int>>> adj;

    Graph(int vertices) : V(vertices), adj(vertices) {}

    void add_edge(int u, int v, int weight) {
        adj[u].push_back(make_pair(v, weight));
        adj[v].push_back(make_pair(u, weight));
    }
};

__global__ void randomPartitionKernel(int *labels, int V, int num_partitions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        float rand_float = curand_uniform(&state);
        labels[idx] = static_cast<int>(rand_float * num_partitions) % num_partitions;
    }
}

__global__ void contiguousPartitionKernel(int *labels, int V, int num_partitions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        labels[idx] = idx % num_partitions;
    }
}

Graph get_graph(int n, int *RowPtr, int* ColIdx, int* weight){
    Graph g(n);
    for(int i=0;i<n;i++){
        int start = RowPtr[i];
        int end = RowPtr[i+1];
        for(int j=start;j<end;j++){
            g.add_edge(i,ColIdx[j],weight[j]);
        }
    }

    return g;
}

int main() {

    int rp[] = {0,2,5,6,8,10};
    int cl[] = {1,2,2,4,5,3,4,6,5,6};
    int w[] = {1,2,3,3,2,2,1,2,1,3};

    Graph g = get_graph(7,rp,cl,w);
    int num_partitions = 3;

    int V = g.V;
    int *random_labels, *contiguous_labels;
    cudaMallocManaged(&random_labels, V * sizeof(int));
    cudaMallocManaged(&contiguous_labels, V * sizeof(int));

    // Launch random partitioning kernel
    int blockSize = 256;
    int numBlocks = (V + blockSize - 1) / blockSize;
    randomPartitionKernel<<<numBlocks, blockSize>>>(random_labels, V, num_partitions);
    cudaDeviceSynchronize();

    // Launch contiguous partitioning kernel
    contiguousPartitionKernel<<<numBlocks, blockSize>>>(contiguous_labels, V, num_partitions);
    cudaDeviceSynchronize();

    // Print random partitioning labels
    cout << "Random Partitioning Labels:";
    for (int i = 0; i < V; ++i) {
        cout << " " << random_labels[i];
    }
    cout << endl;

    // Print contiguous partitioning labels
    cout << "Contiguous Partitioning Labels:";
    for (int i = 0; i < V; ++i) {
        cout << " " << contiguous_labels[i];
    }
    cout << endl;

    // Free allocated memory
    cudaFree(random_labels);
    cudaFree(contiguous_labels);

    return 0;
}
