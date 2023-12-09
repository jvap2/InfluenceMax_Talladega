import pandas as pd
import numpy as np
import networkx as nx

import ndlib.models.ModelConfig as mc
from ndlib.models.epidemics import ThresholdModel
from ndlib.models.epidemics import IndependentCascadesModel
from networkx.algorithms import voterank
import sys

meas_1 = "../../RIM_data/syn/meas.csv"
meas_2 = "../../RIM_data/syn/meas_2.csv"
meas_3 = "../../RIM_data/syn/meas_3.csv"

ver = int(sys.argv[1])

if ver == 1:
    f = meas_1
elif ver == 2:
    f = meas_2
elif ver == 3:
    f = meas_3
else:
    print("Wrong version number")
    sys.exit(1)


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
    df = pd.read_csv("../../Graph_Data_Storage/homo.csv")
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

    g = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph())
    seed_set = pd.read_csv("../../RIM_res/res_4000.csv")
    seeds = seed_set.loc[:,"Seed_Set"].to_numpy()
    print("Seeds:",seeds)
    ## Run several simulations to evaluate the spread
    lt_num_steps = 50
    # Number of nodes in the seed set
    # Determine the model parameter
    lt_threshold = 0.1


    # Run the model
    lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=seeds)
    lt_iterations = lt_model.iteration_bunch(lt_num_steps)


    # Get the number of susceptible, infected and the recovered nodes 
    # in the last step
    print("Final Spread, LT",lt_iterations[-1]["node_count"])

    ic_num_steps = 50
    # Number of nodes in the seed set
    ic_seed_set_size = 100
    # Determine the seed set
    # Determine the model parameter
    ic_threshold = 0.5


    # Run the model
    ic_model_1 = independent_cascade(graph=g, threshold=ic_threshold, seed_set=seeds)
    ic_iterations = ic_model_1.iteration_bunch(ic_num_steps)
    spread_1 = []
    for iteration in ic_iterations:
        spread_1.append(iteration['node_count'][1])
    print("Final Spread, Rand RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])

    percent_lt_spread = lt_iterations[-1]["node_count"][1]/len(g.nodes())
    percent_ic_spread = (ic_iterations[-1]["node_count"][2]+ic_iterations[-1]["node_count"][1])/len(g.nodes())

    exec_data = pd.read_csv(f)
    test_trial=exec_data.shape[0]
    exec_data.loc[test_trial-1, "percent_LT"] = percent_lt_spread
    exec_data.loc[test_trial-1, "percent_IC"] = percent_ic_spread
    exec_data.to_csv(f,index=False)


    vr = voterank(g, number_of_nodes=ic_seed_set_size)
    print("Voterank Nodes:",vr) 


    vr_lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=vr)
    lt_iterations = vr_lt_model.iteration_bunch(lt_num_steps)
    print("Final Spread, LT",lt_iterations[-1]["node_count"])
    # Run the model
    ic_model_2 = independent_cascade(graph=g, threshold=ic_threshold, seed_set=vr)
    ic_iterations = ic_model_2.iteration_bunch(ic_num_steps)
    spread_2 = []
    for iteration in ic_iterations:
        spread_2.append(iteration['node_count'][1])
    print("Final Spread, Voterank RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])


