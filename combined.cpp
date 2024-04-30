#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

class Edge {
public:
    int src, dest, weight;
    Edge(int s, int d, int w) : src(s), dest(d), weight(w) {}
};

class Subset {
public:
    int parent, rank;
};

class Graph {
public:
    int V;
    vector<vector<pair<int, int>>> adj;
    vector<Edge> edges;

    Graph(int vertices, int edges_count) : V(vertices), adj(vertices) {
        edges.reserve(edges_count);
    }

    void add_edge(int u, int v, int weight) {
        edges.emplace_back(u, v, weight);
        adj[u].push_back(make_pair(v, weight));
        adj[v].push_back(make_pair(u, weight));
    }
};

int find(vector<Subset>& subsets, int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
    return subsets[i].parent;
}

void Union(vector<Subset>& subsets, int x, int y) {
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);

    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    else {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

vector<int> kargerMinCut(Graph& graph, int num_partitions) {
    int V = graph.V;
    vector<Edge>& edges = graph.edges;

    vector<Subset> subsets(V);
    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }

    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
    });

    int vertices = V;
    for (const Edge& e : edges) {
        if (vertices <= num_partitions)
            break;

        int subset1 = find(subsets, e.src);
        int subset2 = find(subsets, e.dest);

        if (subset1 != subset2) {
            Union(subsets, subset1, subset2);
            vertices--;
        }
    }

    vector<int> labels(V);
    for (int i = 0; i < V; ++i) {
        int subset_index = find(subsets, i);
        labels[i] = min(subset_index, num_partitions - 1);
    }

    return labels;
}

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

int main() {
    int V = 7;
    int E = 10;
    int num_partitions = 3;

    Graph graph(V, E);
    graph.add_edge(0, 1, 1);
    graph.add_edge(0, 2, 2);
    graph.add_edge(1, 2, 3);
    graph.add_edge(1, 5, 2);
    graph.add_edge(1, 4, 3);
    graph.add_edge(2, 3, 2);
    graph.add_edge(3, 4, 1);
    graph.add_edge(3, 6, 2);
    graph.add_edge(4, 5, 1);
    graph.add_edge(4, 6, 3);

    // Karger Min Cut Partitioning
    vector<int> min_cut_labels = kargerMinCut(graph, num_partitions);
    cout << "Min cut Partitioning Labels: ";
    for (int label : min_cut_labels) {
        cout << label << " ";
    }
    cout << endl;

    // Random partitioning
    vector<int> random_labels = random_partition(graph, num_partitions);
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
