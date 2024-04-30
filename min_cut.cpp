#include <iostream>
#include <vector>
#include <algorithm>

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
    int V, E;
    vector<Edge> edges;
    Graph(int v, int e) : V(v), E(e) {}
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

int main() {
    int V = 7;
    int E = 10;
    int num_partitions = 5;

    Graph graph(V, E);
    graph.edges.push_back(Edge(0, 1, 1));#include <iostream>
#include <vector>
#include <algorithm>

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
    int V, E;
    vector<Edge> edges;
    Graph(int v, int e) : V(v), E(e) {}
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

int main() {
    int V = 7;
    int E = 10;
    int num_partitions = 5;

    Graph graph(V, E);
    graph.edges.push_back(Edge(0, 1, 1));
    graph.edges.push_back(Edge(0, 2, 2));
    graph.edges.push_back(Edge(1, 2, 3));
    graph.edges.push_back(Edge(1, 5, 2));
    graph.edges.push_back(Edge(1, 4, 3));
    graph.edges.push_back(Edge(2, 3, 2));
    graph.edges.push_back(Edge(3, 4, 1));
    graph.edges.push_back(Edge(3, 6, 2));
    graph.edges.push_back(Edge(4, 5, 1));
    graph.edges.push_back(Edge(4, 6, 3));

    vector<int> partition_labels = kargerMinCut(graph, num_partitions);

    cout << "Min cut Partitioning Labels: ";
    for (int i = 0; i < V; ++i) {
        cout << partition_labels[i] << " ";
    }
    cout << endl;

    return 0;
}

    graph.edges.push_back(Edge(0, 2, 2));
    graph.edges.push_back(Edge(1, 2, 3));
    graph.edges.push_back(Edge(1, 5, 2));
    graph.edges.push_back(Edge(1, 4, 3));
    graph.edges.push_back(Edge(2, 3, 2));
    graph.edges.push_back(Edge(3, 4, 1));
    graph.edges.push_back(Edge(3, 6, 2));
    graph.edges.push_back(Edge(4, 5, 1));
    graph.edges.push_back(Edge(4, 6, 3));

    vector<int> partition_labels = kargerMinCut(graph, num_partitions);

    cout << "Min cut Partitioning Labels: ";
    for (int i = 0; i < V; ++i) {
        cout << partition_labels[i] << " ";
    }
    cout << endl;

    return 0;
}
