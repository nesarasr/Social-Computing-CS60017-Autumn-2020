'''
Social Computing CS60017 2020A : Assignment 1
Generate Structure

Submitted by:
Name    : Nesara S R
Roll No : 18IM30014

'''
import snap
import os
import os.path
import sys
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

#Set Random seed to 42
Rnd = snap.TRnd(42)
Rnd.Randomize()




def number_nodes(G):
  node_count = 0
  for NI in G.Nodes():
    node_count+=1
  print("Number of node: %d" %(node_count))

def number_edges(G):

  edge_count = 0
  for EI in G.Edges():
    edge_count+=1
  print("Number of edges: %d" %(edge_count))

def n_degree_nodes(G,n):
  count = snap.CntDegNodes(G, n)
  print("Number of nodes with degree=%d: %d"  %(n,count))

def max_degree_node(G):
  N = snap.GetMxDegNId(G)
  print("Node id(s) with highest degree: %d" %(N))

def plot_degree_dist(G,dataset):
  snap.PlotOutDegDistr(G,"deg_dist_"+dataset , "Degree Distribution")

def full_dia(G,n):
  diam = snap.GetBfsFullDiam(G,n, False)
  print("Approximate full diameter by sampling %d nodes: %d" %(n,diam))
  return diam

def eff_dia(G,n):
  diam = snap.GetBfsEffDiam(G,n, False)
  print("Approximate effective diameter by sampling %d nodes: %.4f" %(n,diam))
  return diam



if __name__=='__main__':
  dataset = sys.argv[1]
  my_path = os.path.abspath(os.path.dirname(__file__))
  path = my_path+'/subgraphs/'+dataset
  graph = snap.LoadEdgeList(snap.PUNGraph,path, 0, 1)
  
  #Question 1
  number_nodes(graph)
  number_edges(graph)
  
  #Question 2
  n_degree_nodes(graph,7)
  max_degree_node(graph)
  
  #Create ../plots directory if it doesn't already exist
  plot_path = my_path+'/plots'
  try:
    os.mkdir(plot_path)
  except:
    pass
  os.chdir(plot_path)
  plot_degree_dist(graph,dataset[:-6])
  os.chdir(my_path)
  
  
  #Question 3
  f_dia=[]
  f_dia.append(full_dia(graph,10))
  f_dia.append(full_dia(graph,100))
  f_dia.append(full_dia(graph,1000))
  print("Approximate full diameter (mean and variance): %.4f, %.4f" %(np.mean(np.array(f_dia)),np.var(np.array(f_dia))))

  eff_d=[]
  eff_d.append(eff_dia(graph,10))
  eff_d.append(eff_dia(graph,100))
  eff_d.append(eff_dia(graph,1000))
  print("Approximate effective diameter (mean and variance): %.4f, %.4f" %(np.mean(np.array(eff_d)),np.var(np.array(eff_d))))
  os.chdir(plot_path)
  snap.PlotShortPathDistr(graph,"shortest_path_"+dataset[:-6], "Shortest Path Distribution")
  os.chdir(my_path)
  

  #Question 4
  print('Fraction of nodes in largest connected component: %.4f' %(snap.GetMxSccSz(graph)))
  EdgeV = snap.TIntPrV()
  snap.GetEdgeBridges(graph, EdgeV)
  count = 0
  for edge in EdgeV:
    count+=1
  print("Number of edge bridges: %d " % count)

  ArtNIdV = snap.TIntV()
  snap.GetArtPoints(graph, ArtNIdV)
  count=0
  for NI in ArtNIdV:
    count+=1
  print("Number of articulation points: %d " % count)
  
  os.chdir(plot_path)
  snap.PlotSccDistr(graph, "connected_comp_"+dataset[:-6], "Strongly connected component size distribution")
  os.chdir(my_path)


  #Question 5
  GraphClustCoeff = snap.GetClustCf (graph, -1)
  print("Average clustering coefficient: %.4f" % GraphClustCoeff) 

  NumTriads = snap.GetTriads(graph,-1)
  print('Number of triads: %d' % NumTriads) 

  node_list=[]
  for NI in graph.Nodes():
    node_list.append(NI.GetId())

  random_node = random.choice(node_list)

  print("Clustering coefficient of random node %d: %.4f " %(random_node,snap.GetNodeClustCf(graph, random_node)))
  print("Number of triads random node %d participates: %d" %(random_node,snap.GetNodeTriads(graph, random_node)))
  print("Number of edges that participate in at least one triad: %d " %(snap.GetTriadEdges(graph)))
  os.chdir(plot_path)
  snap.PlotClustCf(graph,"clustering_coeff_"+dataset[:-6], "Clustering coefficient distribution")


  # Remove .plt and .tab files and retain only .png files in plots
  arr = os.listdir()
  for item in arr:
    if item[-4:]=='.tab' or item[-4:]=='.plt' :
        os.remove(plot_path+'/'+item)
          
  
  

  







