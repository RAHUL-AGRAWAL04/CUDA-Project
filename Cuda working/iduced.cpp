#include <iostream>
#include <vector>

using namespace std;

// Function to get the induced subgraph
void getSubgraph(int n, vector<int>& rowPtr, vector<int>& colIdx, vector<int>& vertices) {
    vector<int> subRowPtr;
    vector<int> subColIdx;

    // Initialize a map to keep track of selected vertices
    vector<bool> isSelected(n, false);
    for (int v : vertices) {
        isSelected[v] = true;
    }

    // Compute the rowPtr for the induced subgraph
    subRowPtr.push_back(0);
    int nnz = 0;
    for (int i = 0; i < n; i++) {
        if (isSelected[i]) {
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                int col = colIdx[j];
                if (isSelected[col]) {
                    subColIdx.push_back(col);
                    nnz++;
                }
            }
            subRowPtr.push_back(nnz);
        }
    }

    // Printing the subgraph
    cout << "Induced Subgraph:\n";
    cout << "rowPtr: ";
    for (int val : subRowPtr) {
        cout << val << " ";
    }
    cout << endl;

    cout << "colIdx: ";
    for (int val : subColIdx) {
        cout << val << " ";
    }
    cout << endl;
}

int main() {
    int n; // Number of vertices
    int m; // Number of edges

    cout << "Enter the number of vertices in the graph: ";
    cin >> n;

    cout << "Enter the number of edges in the graph: ";
    cin >> m;

    vector<int> rowPtr(n + 1); // Row pointers
    vector<int> colIdx(m);      // Column indices

    cout << "Enter the rowPtr array: ";
    for (int i = 0; i <= n; i++) {
        cin >> rowPtr[i];
    }

    cout << "Enter the colIdx array: ";
    for (int i = 0; i < m; i++) {
        cin >> colIdx[i];
    }

    int numVertices;
    cout << "Enter the number of vertices in the induced subgraph: ";
    cin >> numVertices;

    vector<int> vertices(numVertices);
    cout << "Enter the indices of the vertices in the induced subgraph (0-indexed):\n";
    for (int i = 0; i < numVertices; i++) {
        cin >> vertices[i];
    }

    getSubgraph(n, rowPtr, colIdx, vertices);

    return 0;
}
