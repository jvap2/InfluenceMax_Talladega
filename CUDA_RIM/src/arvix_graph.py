import pandas as pd
import numpy as np
import networkx as nx
from check_diff import linear_threshold, independent_cascade
from networkx.algorithms import voterank

File = "../../Graph_Data_Storage/ca-GrQc.txt"

g = nx.read_edgelist(
    File,
    create_using=nx.DiGraph(),
    nodetype=int
)

seed_set = pd.read_csv("../../RIM_res/res_arvix_new.csv")
seeds = seed_set.loc[:,"Seed_Set"].to_numpy()
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

percent_lt_spread = lt_iterations[-1]["node_count"][1]/len(g.nodes())

ic_num_steps = 100
# Number of nodes in the seed set
ic_seed_set_size = 30
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

percent_ic_spread = ic_iterations[-1]["node_count"][1]/len(g.nodes())
print("Percent Spread, LT, IC",percent_lt_spread, percent_ic_spread)

exec_data = pd.read_csv("../../RIM_data/arvix/meas.csv")
test_trial=exec_data.shape[0]
exec_data["percent_LT"].iloc[test_trial-1]=percent_lt_spread
exec_data["percent_IC"].iloc[test_trial-1]=percent_ic_spread
exec_data.to_csv("../../RIM_data/arvix/meas.csv",index=False)


vr = voterank(g, number_of_nodes=ic_seed_set_size)


vr_lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=vr)
lt_iterations = lt_model.iteration_bunch(lt_num_steps)
print("Final Spread VR, LT",lt_iterations[-1]["node_count"])
# Run the model
ic_model_2 = independent_cascade(graph=g, threshold=ic_threshold, seed_set=vr)
ic_iterations = ic_model_2.iteration_bunch(ic_num_steps)
spread_2 = []
for iteration in ic_iterations:
    spread_2.append(iteration['node_count'][1])
print("Final Spread, Voterank RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])

vr_set = set(vr)
seed_set = set(seeds)
print("Intersection:",vr_set.intersection(seed_set))