import networkx as nx
#https://pypi.python.org/pypi/networkx/2.0
import matplotlib.pyplot as plt

FILE1=open("EdgesFileText_names.txt", "rb")
G8=nx.read_edgelist(FILE1, delimiter=",",create_using=nx.Graph(), nodetype=str,data=[("weight", int)])
FILE1.close()
print("G8 is:" ,G8.edges(data=True), "\n\n\n\n")
edge_labels = dict( ((u, v), d["weight"]) for u, v, d in G8.edges(data=True) )
pos = nx.random_layout(G8)
nx.draw(G8, pos, node_color = 'b', edge_labels=edge_labels, with_labels=True)
plt.show()
