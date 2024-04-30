import random
import matplotlib.pyplot as plt
import networkx as nx

class Edge:
    def __init__(self, s, d, w):
        self.src = s
        self.dest = d
        self.weight = w

class Graph:
    def __init__(self, v, e):
        self.V = v
        self.E = e
        self.edge = []

class Subset:
    def __init__(self, p, r):
        self.parent = p
        self.rank = r

def kargerMinCut(graph, num_partitions):
    V = graph.V
    E = graph.E
    edge = graph.edge

    subsets = []
    for v in range(V):
        subsets.append(Subset(v, 0))

    # Sort edges by weight
    sorted_edges = sorted(edge, key=lambda x: x.weight)

    vertices = V
    for e in sorted_edges:
        if vertices <= num_partitions:
            break

        subset1 = find(subsets, e.src)
        subset2 = find(subsets, e.dest)

        if subset1 != subset2:
            Union(subsets, subset1, subset2)
            vertices -= 1

    # Initialize partitions list
    partitions = [[] for _ in range(V)]

    for i in range(V):
        subset_index = find(subsets, i)
        # Ensure that the subset index does not exceed the number of partitions
        subset_index = min(subset_index, num_partitions - 1)
        partitions[subset_index].append(i)

    return partitions

def find(subsets, i):
    if subsets[i].parent != i:
        subsets[i].parent = find(subsets, subsets[i].parent)
    return subsets[i].parent

def Union(subsets, x, y):
    xroot = find(subsets, x)
    yroot = find(subsets, y)

    if subsets[xroot].rank < subsets[yroot].rank:
        subsets[xroot].parent = yroot
    elif subsets[xroot].rank > subsets[yroot].rank:
        subsets[yroot].parent = xroot
    else:
        subsets[yroot].parent = xroot
        subsets[xroot].rank += 1

def main():
    V = 7
    E = 10
    num_partitions = 3
    graph = Graph(V, E)
    graph.edge.append(Edge(0, 1, 1))
    graph.edge.append(Edge(0, 2, 2))
    graph.edge.append(Edge(1, 2, 3))
    graph.edge.append(Edge(1, 5, 2))
    graph.edge.append(Edge(1, 4, 3))
    graph.edge.append(Edge(2, 3, 2))
    graph.edge.append(Edge(3, 4, 1))
    graph.edge.append(Edge(3, 6, 2))
    graph.edge.append(Edge(4, 5, 1))
    graph.edge.append(Edge(4, 6, 3))

    G = nx.Graph()
    G.add_weighted_edges_from([(e.src, e.dest, e.weight) for e in graph.edge])

    plt.figure(figsize=(8, 6))
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.title('Original Graph')

    partitions = kargerMinCut(graph, num_partitions)

    G_partition = nx.Graph()
    for i, partition in enumerate(partitions):
        for node in partition:
            G_partition.add_node(node, partition=i+1)
    G_partition.add_weighted_edges_from([(e.src, e.dest, e.weight) for e in graph.edge])

    plt.subplot(122)
    pos = nx.spring_layout(G_partition)
    colors = [plt.cm.tab10(i % 10) for i in range(num_partitions)]
    node_colors = [colors[G_partition.nodes[node]['partition'] - 1] for node in G_partition.nodes]
    edge_labels = {(e.src, e.dest): e.weight for e in graph.edge}
    nx.draw(G_partition, pos, with_labels=True, font_weight='bold', node_color=node_colors)
    nx.draw_networkx_edge_labels(G_partition, pos, edge_labels=edge_labels)
    plt.title('Partitioned Graph')
    plt.show()

if __name__ == '__main__':
    main()
