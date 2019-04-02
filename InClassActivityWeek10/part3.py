import networkx as nx
import matplotlib.pyplot as plt

def main():

    with open("dataset.txt", "rb") as file:
        G = nx.read_edgelist(file, delimiter = ',', create_using = nx.Graph(), nodetype = str, data = [("weight", int)])

    print("Graph is", G.edges(data = True))
    edge_labels = dict( ((z1, z2), z["weight"]) for z1, z2, z in G.edges(data = True) )
    pos = nx.random_layout(G)
    nx.draw(G, pos, edge_labels = edge_labels, with_labels = True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
    plt.show()
    
main()
