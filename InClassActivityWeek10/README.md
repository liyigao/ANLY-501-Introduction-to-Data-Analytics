# In Class Activity Week 10 - Network Analysis

## Part 1

Read `EdgesFileText_names.txt` data. Change the colors of the nodes in the network to blue.

## Part 2

**Task:**  
Conduct a small network analysis using the Karate Club network.  
**Network Details:**  
Zachary's Karate Club graph  
Data file from: http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm  
Direct Link to data (the data is an adjacency matrix that is non-weighted)  
http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/zachary.dat  
**Reference:**  
Zachary W. (1977). An information flow model for conflict and fission in small groups. Journal of Anthropological Research, 33, 452-473.  
These are data collected from the members of a university karate club by Wayne Zachary. A node is a member and an edge indicates that the two members interact. Zachary (1977) used these data and an information flow model of network conflict resolution to explain the split-up of this group following disputes among the members.  
**Instruction:**  
Create a plot that colors nodes based on degree. Hint: (1) similar to betweeness centrality, compute the degree centrality, and (2) assign the values of degree to the plot, similar to the cluster colors.

## Part 3

1)	Create (by hand) a small dataset (as `dataset.txt`) of edges and edge weights. Have 10 to 20 rows. All student datasets will be different.
2)	Use NetworkX to read in the dataset and create a network. 
3)	The network should have labeled nodes.
4)	Extra credit: Label the edges with the weights from your dataset.
