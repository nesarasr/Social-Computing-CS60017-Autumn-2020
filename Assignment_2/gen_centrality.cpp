// gen_centrality.cpp does the following tasks

// 1) Compute closeness centrality for each nodes and output a .txt file containing the centrality values in non-increasing order

// 2) Compute betweenness centrality for each nodes and output a .txt file containing the centrality values in non-increasing order

// 3) Compute pagerank centrality for each nodes with a teleportation bias towards nodeIDs that are divisible by 4 and output a .txt file containing the centrality values in non-increasing order


//NAME : NESARA S R

//ROLL NO : 18IM30014
#pragma GCC optimize("Ofast")
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "stdc++.hpp" //using instead of #include <bits/stdc++.h>

using namespace std;


//structure to represent a graph node
struct AdjListNode
{
    int dest;
    int weight;
    struct AdjListNode* next;
};

//structure to represent an adjacency list
struct AdjList
{
    struct AdjListNode *head; // pointer to head node of list
};

// A structure to represent a graph. A graph is an array of adjacency lists.
// Size of array will be V (number of vertices in graph)
struct Graph
{
    int V;
    struct AdjList* array;
};

// function to create a new adjacency list node
struct AdjListNode* newAdjListNode(int dest, int weight)
{
    struct AdjListNode* newNode =
            (struct AdjListNode*) malloc(sizeof(struct AdjListNode));
    newNode->dest = dest;
    newNode->weight = weight;
    newNode->next = NULL;
    return newNode;
}

// function that creates a graph of V vertices
struct Graph* createGraph(int V)
{
    struct Graph* graph = (struct Graph*) malloc(sizeof(struct Graph));
    graph->V = V;

    // Create an array of adjacency lists. Size of array will be V
    graph->array = (struct AdjList*) malloc(V * sizeof(struct AdjList));

    // Initialize each adjacency list as empty by making head as NULL
    for (int i = 0; i < V; ++i)
        graph->array[i].head = NULL;

    return graph;
}

// Adds an edge to an undirected graph
void addEdge(struct Graph* graph, int src, int dest, int weight)
{
    // Add an edge from src to dest. A new node is added to the adjacency
    // list of src. The node is added at the beginning
    struct AdjListNode* newNode = newAdjListNode(dest, weight);
    newNode->next = graph->array[src].head;
    graph->array[src].head = newNode;

    // Since graph is undirected, add an edge from dest to src also
    newNode = newAdjListNode(src, weight);
    newNode->next = graph->array[dest].head;
    graph->array[dest].head = newNode;
}

// Structure to represent a min heap node
struct MinHeapNode
{
    int v;
    int dist;
};

// Structure to represent a min heap
struct MinHeap
{
    int size;     // Number of heap nodes present currently
    int capacity; // Capacity of min heap
    int *pos;     // This is needed for decreaseKey()
    struct MinHeapNode **array;
};

// A utility function to create a new Min Heap Node
struct MinHeapNode* newMinHeapNode(int v, int dist)
{
    struct MinHeapNode* minHeapNode =
        (struct MinHeapNode*) malloc(sizeof(struct MinHeapNode));
    minHeapNode->v = v;
    minHeapNode->dist = dist;
    return minHeapNode;
}

// A utility function to create a Min Heap
struct MinHeap* createMinHeap(int capacity)
{
    struct MinHeap* minHeap =
        (struct MinHeap*) malloc(sizeof(struct MinHeap));
    minHeap->pos = (int *)malloc(capacity * sizeof(int));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array =
        (struct MinHeapNode**) malloc(capacity * sizeof(struct MinHeapNode*));
    return minHeap;
}

// A utility function to swap two nodes of min heap. Needed for min heapify
void swapMinHeapNode(struct MinHeapNode** a, struct MinHeapNode** b)
{
    struct MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

// A standard function to heapify at given idx
// This function also updates position of nodes when they are swapped.
// Position is needed for decreaseKey()
void minHeapify(struct MinHeap* minHeap, int idx)
{
    int smallest, left, right;
    smallest = idx;
    left = 2 * idx + 1;
    right = 2 * idx + 2;

    if (left < minHeap->size &&
        minHeap->array[left]->dist < minHeap->array[smallest]->dist )
    smallest = left;

    if (right < minHeap->size &&
        minHeap->array[right]->dist < minHeap->array[smallest]->dist )
    smallest = right;

    if (smallest != idx)
    {
        // The nodes to be swapped in min heap
        MinHeapNode *smallestNode = minHeap->array[smallest];
        MinHeapNode *idxNode = minHeap->array[idx];

        // Swap positions
        minHeap->pos[smallestNode->v] = idx;
        minHeap->pos[idxNode->v] = smallest;

        // Swap nodes
        swapMinHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);

        minHeapify(minHeap, smallest);
    }
}

// A utility function to check if the given minHeap is empty or not
int isEmpty(struct MinHeap* minHeap)
{
    return minHeap->size == 0;
}

// Standard function to extract minimum node from heap
struct MinHeapNode* extractMin(struct MinHeap* minHeap)
{
    if (isEmpty(minHeap))
        return NULL;

    // Store the root node
    struct MinHeapNode* root = minHeap->array[0];

    // Replace root node with last node
    struct MinHeapNode* lastNode = minHeap->array[minHeap->size - 1];
    minHeap->array[0] = lastNode;

    // Update position of last node
    minHeap->pos[root->v] = minHeap->size-1;
    minHeap->pos[lastNode->v] = 0;

    // Reduce heap size and heapify root
    --minHeap->size;
    minHeapify(minHeap, 0);

    return root;
}

// Function to decrease dist value of a given vertex v. This function
// uses pos[] of min heap to get the current index of node in min heap
void decreaseKey(struct MinHeap* minHeap, int v, int dist)
{
    // Get the index of v in heap array
    int i = minHeap->pos[v];

    // Get the node and update its dist value
    minHeap->array[i]->dist = dist;

    // Travel up while the complete tree is not hepified.
    // This is a O(Logn) loop
    while (i && minHeap->array[i]->dist < minHeap->array[(i - 1) / 2]->dist)
    {
        // Swap this node with its parent
        minHeap->pos[minHeap->array[i]->v] = (i-1)/2;
        minHeap->pos[minHeap->array[(i-1)/2]->v] = i;
        swapMinHeapNode(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);

        // move to parent index
        i = (i - 1) / 2;
    }
}

// A utility function to check if a given vertex
// 'v' is in min heap or not
bool isInMinHeap(struct MinHeap *minHeap, int v)
{
if (minHeap->pos[v] < minHeap->size)
    return true;
return false;
}

// The main function that calulates distances of shortest paths from src to all
// vertices. It is a O(ELogV) function
vector <int> shortest_path(struct Graph* graph, int src)
{
    int V = graph->V;// Get the number of vertices in graph
    vector <int> dist(V,0);     // dist values used to pick minimum weight edge in cut

    // minHeap represents set E
    struct MinHeap* minHeap = createMinHeap(V);

    // Initialize min heap with all vertices. dist value of all vertices
    for (int v = 0; v < V; ++v)
    {
        dist[v] = INT_MAX;
        minHeap->array[v] = newMinHeapNode(v, dist[v]);
        minHeap->pos[v] = v;
    }

    // Make dist value of src vertex as 0 so that it is extracted first
    minHeap->array[src] = newMinHeapNode(src, dist[src]);
    minHeap->pos[src] = src;
    dist[src] = 0;
    decreaseKey(minHeap, src, dist[src]);

    // Initially size of min heap is equal to V
    minHeap->size = V;

    // In the followin loop, min heap contains all nodes
    // whose shortest distance is not yet finalized.
    while (!isEmpty(minHeap))
    {
        // Extract the vertex with minimum distance value
        struct MinHeapNode* minHeapNode = extractMin(minHeap);
        int u = minHeapNode->v; // Store the extracted vertex number

        // Traverse through all adjacent vertices of u (the extracted
        // vertex) and update their distance values
        struct AdjListNode* pCrawl = graph->array[u].head;
        while (pCrawl != NULL)
        {
            int v = pCrawl->dest;

            // If shortest distance to v is not finalized yet, and distance to v
            // through u is less than its previously calculated distance
            if (isInMinHeap(minHeap, v) && dist[u] != INT_MAX &&
                                        pCrawl->weight + dist[u] < dist[v])
            {
                dist[v] = dist[u] + pCrawl->weight;

                // update distance value in min heap also
                decreaseKey(minHeap, v, dist[v]);
            }
            pCrawl = pCrawl->next;
        }
    }

    // print the calculated shortest distances
    //cout << setprecision(6);
    return dist;
}


// CLOSENESS CENTRALITY COMPUTED HERE
float closeness(struct Graph* graph, int src, int n)
{
    //printf("Vertex Distance from Source\n");
    vector<int> dist = shortest_path(graph,src);
    int sum = 0.0;
    for (int i = 0; i < n; ++i)
        //printf("%d \t\t %d\n", i, dist[i]);
        sum+=dist[i];
    return (float(n-1)/float(sum));
}

//STL implementation of graph
void add_edge(vector<int> adj[],
              int src, int dest)
{
    adj[src].push_back(dest);
    adj[dest].push_back(src);
}


//BETWEENNESS CENTRALITY COMPUTED HERE
// Computed using Brandes Algorithm, ref : 'http://www.uvm.edu/pdodds/research/papers/others/2001/brandes2001a.pdf'


vector <float> brandes(vector<int> adj[],int n)
{
    vector<float> C_b(n,0);
    for(int s=0;s<n;s++)
    {
        
        stack<int> S;
        vector <vector <int> > P(n);
        vector<int> sigma(n,0);
        sigma[s]=1;
        vector<int> d(n,-INT_MAX);
        d[s]=0;
        queue<int> Q;
        Q.push(s);
        int v;
        int w;
        while(!Q.empty())
        {
            v=Q.front();
            Q.pop();
            S.push(v);
            for(int i=0;i<adj[v].size();i++)
            {
                w = adj[v][i];
                if(d[w]<0)
                {
                    Q.push(w);
                    d[w] = d[v]+1;
                }
                if(d[w]==d[v]+1)
                {
                    sigma[w]=sigma[w]+sigma[v];
                    P[w].push_back(v);
                }
                
            }
        }
        vector<float> delta(n,0.0);
        while(!S.empty())
        {
            w=S.top();
            S.pop();
            for(int i=0;i<P[w].size();i++)
            {
                v = P[w][i];
                delta[v] = delta[v]+ (float(sigma[v])/float(sigma[w]))*(1+delta[w]);
                if(w!=s)
                {
                     C_b[w] = C_b[w] + delta[w];
                }
            }
        }
    }
    
    return C_b;
}

//function that normalizes pagerank values after every iteration : L2 Normalization
vector <float> normalize(vector<float> pr,int n)
{
    float sum = 0.0;
    for(int i=0;i<pr.size();i++)
    {
        sum+= (pr[i]*pr[i]);
    }
    
    sum = sqrt(sum);
    
    for(int i=0;i<pr.size();i++)
    {
        pr[i]/=sum;
    }
    return pr;
}


//PAGERANK COMPUTED HERE
vector <float> pagerank(vector<int> adj[],int n)
{
    float d_u = 0.0009900990099009901;   //bias vector
    vector<float> d(n,0);
    for(int i=0;i<n;i++)
    {
        if(i%4==0)   //biased towards node ids that are perfectly divisible by 4
        {
            d[i]=d_u;
        }
    }
    vector<float> pr = d;  //pagerank initialization
    
    float theta = 0.0000001;  //1e-7
    float delta;  //measures max change in values between iterations
    while(1)
    {
        delta = 0;
        for(int i=0;i<n;i++)
        {
            float pr_copy = pr[i];
            float t=0.0;
            for(int j=0;j<adj[i].size();j++)
            {
                t+= (pr[adj[i][j]] / adj[adj[i][j]].size());
            }
            pr[i] = 0.8*t + 0.2*d[i];
            delta = max(delta,abs(pr_copy-pr[i]));
            
        }
        if(delta<theta)
        {
            break;
        }
    }
    
    return pr;

    
}


// MAIN
int main()
{
       
    // create the graph from "facebook_combined.txt"
    
    int V = 4039;                   //number of nodes
    vector <int> adj[V];
    struct Graph* graph = createGraph(V);
    string line;
    int n1,n2;
    ifstream myfile ("facebook_combined.txt");
    if (myfile.is_open())
    {
      while ( getline (myfile,line) )
      {
        myfile >> n1 >> n2;
          addEdge(graph, n1, n2,1);
          add_edge(adj, n1, n2);
      }
      myfile.close();
    }
    
    vector<pair<float, int> > cc;
    for(int i=0;i<V;i++)
    {
        float x = closeness(graph,i,V);   //compute closeness centrality for every node 'i'
        cc.push_back(make_pair(x, i));
    }
    sort(cc.rbegin(), cc.rend());      //sort the closeness centrality list
    ofstream ccfile ("./centralities/closeness.txt");
    if (ccfile.is_open())
    {
        for (int i = 0; i <V; i++)
        {
            ccfile<< cc[i].second << "\t";
            ccfile<<fixed<<setprecision(6)<< cc[i].first << endl;    //save sorted values in "closeness.txt" with precision upto 6 decimal places
            
        }
            ccfile.close();
    }
    else cout << "Unable to open file";
    
    vector <float> y = brandes(adj,V);   //compute betweenness centrality
    vector<pair<float, int> > bc;
    for(int i=0;i<V;i++)
    {
        bc.push_back(make_pair(y[i]/(2*(V-1)*(V-2)), i));   //save normalized values of betweenness centrality
    }
    sort(bc.rbegin(), bc.rend());      //sort the betweenness centrality list

    ofstream bcfile ("./centralities/betweenness.txt");
    if (bcfile.is_open())
    {
        for (int i = 0; i <V; i++)
        {
            bcfile<< bc[i].second << "\t";
            bcfile<<fixed<<setprecision(6)<< bc[i].first << endl;  //save sorted values in "betweenness.txt" with precision upto 6 decimal places
            
        }
            bcfile.close();
    }
    else cout << "Unable to open file";
    
    
    vector<float> z = pagerank(adj,V);      //compute pagerank centrality
    vector<pair<float, int> > pr;
    for(int i=0;i<V;i++)
    {
        pr.push_back(make_pair(z[i], i));
    }
    sort(pr.rbegin(), pr.rend());      //sort the pagerank centrality list
    
    ofstream prfile ("./centralities/pagerank.txt");
    if (prfile.is_open())
    {
        for (int i = 0; i <V; i++)
        {
            prfile<< pr[i].second << "\t";
            prfile<<fixed<<setprecision(6)<< pr[i].first << endl;  //save sorted values in "pagerank.txt" with precision upto 6 decimal
            
        }
            prfile.close();
    }
    else cout << "Unable to open file";
    ;

    return 0;
}

