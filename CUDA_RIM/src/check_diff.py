import pandas as pd
import numpy as np
import networkx as nx

import ndlib.models.ModelConfig as mc
from ndlib.models.epidemics import ThresholdModel
from ndlib.models.epidemics import IndependentCascadesModel



def linear_threshold(graph, threshold, seed_set):
    # Model selection
    model = ThresholdModel(graph)
    
    # Model configuration
    config = mc.Configuration()
    ## Set edge parameters
    for edge in graph.edges():
        config.add_edge_configuration("threshold", edge, threshold)        
    ## Set the initial infected nodes
    config.add_model_initial_configuration("Infected", seed_set)
    
    # Set the all configuations
    model.set_initial_status(config)
    return model


def independent_cascade(graph, threshold, seed_set):
    """
    The model performing independent cascade simulation
    """
    # Model selection
    model = IndependentCascadesModel(graph)
    
    # Model configuration
    config = mc.Configuration()
    ## Set edge parameters
    for edge in graph.edges():
        config.add_edge_configuration("threshold", edge, threshold)        
    ## Set the initial infected nodes
    config.add_model_initial_configuration("Infected", seed_set)
    
    # Set the all configuations
    model.set_initial_status(config)
    return model


if __name__ == "__main__":
    df = pd.read_csv("../Graph_Data_Storage/homo.csv")
    src=df.loc[:,"source"].to_numpy()
    dst=df.loc[:,"target"].to_numpy()
    df_info = pd.read_csv("../../Graph_Data_Storage/homo_info.csv")
    no_nodes = df_info.loc[:,"No. Nodes"].to_numpy()
    no_nodes = no_nodes[0]
    print(no_nodes)
    adj_mat = np.zeros(shape=(no_nodes,no_nodes))
    k=50
    for s,d in zip(src,dst):
        adj_mat[s][d]=1

