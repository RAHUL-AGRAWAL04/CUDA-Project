#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

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

vector<int> random_partition(const Graph& graph, int num_partitions) {
    vector<int> labels(graph.V);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, num_partitions - 1);

    for (int v = 0; v < graph.V; ++v) {
        labels[v] = dis(gen);
    }
    return labels;
}

vector<int> contiguous_partition(const Graph& graph, int num_partitions) {
    vector<int> labels(graph.V);
    for (int v = 0; v < graph.V; ++v) {
        labels[v] = v % num_partitions;
    }
    return labels;
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

    // Random partitioning
    vector<int> random_labels = random_partition(g, num_partitions);
    cout << "Random Partitioning Labels:";
    for (int label : random_labels) {
        cout << " " << label;
    }
    cout << endl;

    // Contiguous partitioning
    vector<int> contiguous_labels = contiguous_partition(graph, num_partitions);
    cout << "Contiguous Partitioning Labels:";
    for (int label : contiguous_labels) {
        cout << " " << label;
    }
    cout << endl;

    return 0;
}
