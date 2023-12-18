import pandas as pd
import numpy as np
import networkx as nx
from check_diff import linear_threshold, independent_cascade
from networkx.algorithms import voterank
import sys

meas_1="../../RIM_data/HepTh/meas.csv"
meas_2="../../RIM_data/HepTh/meas_2.csv"
meas_3="../../RIM_data/HepTh/meas_3.csv"

ver = int(sys.argv[1])
k = int(sys.argv[2])

if ver == 1:
    f = meas_1
elif ver == 2:
    f = meas_2
elif ver == 3:
    f = meas_3
else:
    print("Wrong version number")
    sys.exit(1)

File = "../../Graph_Data_Storage/ca-HepTh.txt"

g = nx.read_edgelist(
    File,
    create_using=nx.DiGraph(),
    nodetype=int
)

seed_set = pd.read_csv("../../RIM_res/res_HepTh_new.csv")
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

ic_num_steps = 50
# Number of nodes in the seed set
ic_seed_set_size = k
# Determine the seed set
# Determine the model parameter
ic_threshold = 0.5


# Run the model
ic_model_1 = independent_cascade(graph=g, threshold=ic_threshold, seed_set=seeds)
ic_iterations = ic_model_1.iteration_bunch(ic_num_steps)
spread_1 = []
for iteration in ic_iterations:
    spread_1.append(iteration['node_count'][1])
print("IC Final Spread, Rand RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])

percent_lt_spread = lt_iterations[-1]["node_count"][1]/len(g.nodes())
percent_ic_spread = (ic_iterations[-1]["node_count"][2]+ic_iterations[-1]["node_count"][1])/len(g.nodes())

exec_data = pd.read_csv(f)
test_trial=exec_data.shape[0]
exec_data.loc[test_trial-1, "percent_LT"] = percent_lt_spread
exec_data.loc[test_trial-1, "percent_IC"] = percent_ic_spread
exec_data.to_csv(f,index=False)


vr = voterank(g, number_of_nodes=ic_seed_set_size)


vr_lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=vr)
lt_iterations = vr_lt_model.iteration_bunch(lt_num_steps)
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

curip_seeds_lt = pd.read_csv("../../RIM_res/curip_hepth_LT_new.csv")
curip_seeds_lt = curip_seeds_lt.loc[:,"Seed_Set"].to_numpy()
## Run several simulations to evaluate the spread
curip_seeds_ic = pd.read_csv("../../RIM_res/curip_hepth_IC_new.csv")
curip_seeds_ic = curip_seeds_ic.loc[:,"Seed_Set"].to_numpy()
## Run several simulations to evaluate the spread
ic_model_curip = independent_cascade(graph=g, threshold=ic_threshold, seed_set=curip_seeds_ic)
ic_iterations = ic_model_curip.iteration_bunch(ic_num_steps)
spread_1 = []
for iteration in ic_iterations:
    spread_1.append(iteration['node_count'][1])
print("Final Spread, Curip RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])
percent_ic_spread = (ic_iterations[-1]["node_count"][2]+ic_iterations[-1]["node_count"][1])/len(g.nodes())

lt_model_curip = linear_threshold(graph=g, threshold=lt_threshold, seed_set=curip_seeds_lt)
lt_iterations = lt_model_curip.iteration_bunch(lt_num_steps)
percent_lt_spread = lt_iterations[-1]["node_count"][1]/len(g.nodes())
print("Final Spread, Curip RIM, susceptible, infected and the recovered nodes ",lt_iterations[-1]["node_count"])


exec_data = pd.read_csv(f)
test_trial=exec_data.shape[0]
exec_data.loc[test_trial-1, "percent_LT_CU"] = percent_lt_spread
exec_data.loc[test_trial-1, "percent_IC_CU"] = percent_ic_spread
exec_data.to_csv(f,index=False)
