import pandas as pd
import numpy as np
import networkx as nx

File = "../Graph_Data_Storage/web-NotreDame.txt"
top = pd.read_csv("res_nd.csv").to_numpy()

g = nx.read_edgelist(
    File,
    create_using=nx.DiGraph(),
    nodetype=int
)

def average_distance(graph, nodes):
    distances = []
    for node in nodes:
        lengths = nx.shortest_path_length(graph, source=node)
        distances.extend([lengths[target] for target in nodes if target in lengths])
    return np.mean(distances) if distances else 0

d = average_distance(g, top)