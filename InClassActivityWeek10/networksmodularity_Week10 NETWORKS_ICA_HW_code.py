#######################################
# In-class Activity and HW
#
# Network analysis
#
# We are using the networkx library
# It is already installed with Anaconda
# Documentation:
# https://networkx.readthedocs.io/en/stable/
#
# Also using community library
# Installation directions:
# https://bitbucket.org/taynaud/python-louvain
## Choose downloads image along the left of the page to download
##
##to install on anaconda - use cmd command line
## move all portions of the folder to you python scripts area
## cd to where you can see setup.py in folder then type
## python setup.py install
## on the command line to install
## restart spyder and then run code below
#######################################


import networkx as nx
import matplotlib.pyplot as plt
import community
from community import community_louvain

myNetXGraph=nx.karate_club_graph()

# Prints summary information about the graph
print(nx.info(myNetXGraph))

# Print the degree of each node
print("Node Degree")
for v in myNetXGraph:
    print('%s %s' % (v,myNetXGraph.degree(v)))

# Computer and print other stats    
nbr_nodes = nx.number_of_nodes(myNetXGraph)
nbr_edges = nx.number_of_edges(myNetXGraph)
nbr_components = nx.number_connected_components(myNetXGraph)

print("Number of nodes:", nbr_nodes)
print("Number of edges:", nbr_edges)
print("Number of connected components:", nbr_components)
print("Density:", nbr_edges/(nbr_nodes*(nbr_nodes-1)/2))


# Draw the network using the default settings
nx.draw(myNetXGraph)
plt.clf()

# Draw, but change the color to to blue
nx.draw(myNetXGraph, node_color="blue")
plt.clf()

# Compute betweeness and then store the value with each node in the networkx graph
betweenList = nx.betweenness_centrality(myNetXGraph)
isinstance(betweenList, dict)
nx.set_node_attributes(myNetXGraph, betweenList, "betweenness")
print()
print("Betweeness of each node")
print(betweenList)

#####################
# Clustering
# Code from: http://perso.crans.org/aynaud/communities/#
#####################
# Conduct modularity clustering
partition = community_louvain.best_partition(myNetXGraph)

# Print clusters (You will get a list of each node with the cluster you are in)
print()
print("Clusters")
print(partition)

# Setup colors and graph layout. Then print.
size = float(len(set(partition.values())))
pos = nx.spring_layout(myNetXGraph)
count = 0.
for com in set(partition.values()) :
     count += 1.
     list_nodes = [nodes for nodes in partition.keys()
                                 if partition[nodes] == com]
     nx.draw_networkx_nodes(myNetXGraph, pos, list_nodes, node_size = 50,
                                node_color = str(count / size))
nx.draw_networkx_edges(myNetXGraph,pos, alpha=0.5)
plt.show()

# Compute degree
degreeList = nx.degree_centrality(myNetXGraph)
nx.set_node_attributes(myNetXGraph, degreeList, "degree")
print()
print("Degree of each node")
print(degreeList)
degreeValues = [degreeList.get(node, 0.5) for node in myNetXGraph.nodes()]
nx.draw(myNetXGraph, cmap = plt.get_cmap('jet'), node_color = degreeValues)
plt.show()
