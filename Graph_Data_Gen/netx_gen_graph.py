import networkx as nx
import sys
import random
import numpy as np
import os
import polars as pl
import pandas as pd


folder=os.getcwd()

num_node = int(sys.argv[1])
m = int(sys.argv[2])
p = float(sys.argv[3])
seed = int(sys.argv[4])

G=nx.powerlaw_cluster_graph(num_node, m, p, seed=seed)
num_edges=G.number_of_edges()
print(num_edges)
df_g=nx.to_pandas_edgelist(G)
print(df_g.head())
df_g = df_g.sort_values(by=['target','source'])
print(df_g.head())
df_g.to_csv("../Graph_Data_Storage/homo.csv", index=False)

Network_Info = open("../Graph_Data_Storage/homo_info.csv", 'w')
Network_Info.write("No. Nodes,No. Edges \n")
Network_Info.write(str(sys.argv[1]) + "," + str(num_edges) + "\n")

Network_Info.close()