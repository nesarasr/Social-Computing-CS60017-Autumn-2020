'''
analyze_centrality.py does the following tasks:

1) Compute closeness centrality for each node using snap.GetClosenessCentr and check for number of intersecting values in the top 100 with that of closeness centrality list generated by gen_centrality.cpp

2) Compute betweenness centrality for each node using snap.GetBetweennessCentr and check for number of intersecting values in the top 100 with that of betweenness centrality list generated by gen_centrality.cpp

3) Compute pagerank centrality for each node using snap.GetPageRank and check for number of intersecting values in the top 100 with that of pagerank centrality list generated by gen_centrality.cpp


Submitted by :

Name : Nesara S R

Roll No : 18IM30014



'''


import snap
import os
import os.path
import sys
import numpy as np
import random
import warnings
import pathlib
warnings.filterwarnings("ignore")


import pathlib
root_path = str(pathlib.Path().absolute())

path = root_path+'/facebook_combined.txt'
graph = snap.LoadEdgeList(snap.PUNGraph,path, 0, 1)

num_nodes=0
for N in graph.Nodes():
    x=N.GetId()
    num_nodes+=1


# COMPUTE CLOSENESS CENTRALITY
cc = np.zeros(num_nodes)
for NI in graph.Nodes():
    CloseCentr = snap.GetClosenessCentr(graph, NI.GetId())
    cc[NI.GetId()] = CloseCentr


file_handle = open(root_path+'/centralities/closeness.txt', 'r')
lines_list = file_handle.readlines()

closeness_cpp = []
for i in range(len(lines_list)):
    cols = list(val for val in lines_list[i].split())[0]
    closeness_cpp.append(cols)

# TOP 100 LIST OBTAINED FROM gen_centrality.cpp
closeness_cpp = closeness_cpp[:100]

closeness_py = sorted(range(len(cc)), key=lambda k: cc[k])[-100:]

# COMPUTE INTERSECTION SET
intersection_closeness = np.intersect1d(np.array(closeness_py),np.array(closeness_cpp))

# PRINT SIZE OF INTERSECTION SET i.e NUMBER OF OVERLAPS
print('#overlaps for Closeness Centrality:',len(intersection_closeness))

# BETWEENNESS CENTRALITY
Nodes = snap.TIntFltH()
Edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(graph, Nodes, Edges,0.8)  # SET nodeFrac TO 0.8
bc = np.zeros(num_nodes)
for node in Nodes:
    bc[node] = Nodes[node]*2/((num_nodes-1)*(num_nodes-2))
    
betweenness_py = sorted(range(len(bc)), key=lambda k: bc[k])[-100:]

file_handle = open(root_path+'/centralities/betweenness.txt', 'r')
lines_list = file_handle.readlines()

betweenness_cpp = []
for i in range(len(lines_list)):
    cols = list(val for val in lines_list[i].split())[0]
    betweenness_cpp.append(cols)
    
    
# TOP 100 LIST OBTAINED FROM gen_centrality.cpp
betweenness_cpp = betweenness_cpp[:100]

# COMPUTE INTERSECTION SET
intersection_betweenness = np.intersect1d(np.array(betweenness_cpp),np.array(betweenness_py))

# PRINT SIZE OF INTERSECTION SET i.e NUMBER OF OVERLAPS
print('#overlaps for Betweenness Centrality:',len(intersection_betweenness))


# PAGERANK CENTRALITY
PRankH = snap.TIntFltH()
snap.GetPageRank(graph, PRankH, 0.8, 1e-7,100)  # SET alpha TO 0.8 AND 
pr = np.zeros(4039)
for item in PRankH:
    pr[item] = PRankH[item]
    
pagerank_py = sorted(range(len(pr)), key=lambda k: pr[k])[-100:]

file_handle = open(root_path+'/centralities/pagerank.txt', 'r')
lines_list = file_handle.readlines()

pagerank_cpp = []
for i in range(len(lines_list)):
    cols = list(val for val in lines_list[i].split())[0]
    pagerank_cpp.append(cols)
    
    
# TOP 100 LIST OBTAINED FROM gen_centrality.cpp
pagerank_cpp = pagerank_cpp[:100]

# COMPUTE INTERSECTION SET
intersection_pagerank = np.intersect1d(np.array(pagerank_cpp),np.array(pagerank_py))

# PRINT SIZE OF INTERSECTION SET i.e NUMBER OF OVERLAPS
print('#overlaps for PageRank Centrality:',len(intersection_pagerank))