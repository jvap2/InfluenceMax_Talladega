import pandas as pd
import numpy as np
import networkx as nx
from check_diff import linear_threshold, independent_cascade
from networkx.algorithms import voterank
import sys


meas_1="../../RIM_data/wiki-vote/meas.csv"
meas_2 = "../../RIM_data/wiki-vote/meas_2.csv"
meas_3 = "../../RIM_data/wiki-vote/meas_3.csv"
meas_4 = "../../RIM_data/wiki-vote/meas_4.csv"
meas_5 = "../../RIM_data/wiki-vote/meas_5.csv"
meas_6 = "../../RIM_data/wiki-vote/meas_6.csv"
meas_7 = "../../RIM_data/wiki-vote/meas_7.csv"

ver = int(sys.argv[1])
k = int(sys.argv[2])
direct = str(sys.argv[3])

if ver == 1:
    f = meas_1
elif ver == 2:
    f = meas_2
elif ver == 3:
    f = meas_3
elif ver == 4:
    f = meas_4
elif ver == 5:
    f = meas_5
elif ver == 6:
    f = meas_6
elif ver == 7:
    f = meas_7
else:
    print("Wrong version number")
    sys.exit(1)

File = "../../Graph_Data_Storage/wiki-Vote.txt"

g = nx.read_edgelist(
    File,
    create_using=nx.DiGraph(),
    nodetype=int
)

seed_set = pd.read_csv("../../RIM_res/res_wiki_new.csv")
seeds = seed_set.loc[:,"Seed_Set"].to_numpy()
## Run several simulations to evaluate the spread
lt_num_steps = 50
# Number of nodes in the seed set
# Determine the model parameter
lt_threshold = 0.1
no_nodes = g.number_of_nodes()
rand_chance = np.arange(no_nodes)
np.random.shuffle(rand_chance)


ic_seed_set_size = k
rand_seeds = rand_chance[:ic_seed_set_size]

# Run the model
lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=seeds)
lt_iterations = lt_model.iteration_bunch(lt_num_steps)

rand_lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=rand_seeds)
rand_lt_iterations = rand_lt_model.iteration_bunch(lt_num_steps)
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
rand_ic_model = independent_cascade(graph=g, threshold=ic_threshold, seed_set=rand_seeds)   
rand_ic_iterations = rand_ic_model.iteration_bunch(ic_num_steps)
spread_1 = []
spread_2 = []
for iteration in ic_iterations:
    spread_1.append(iteration['node_count'][1])
print("Final Spread, Rand RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])
for iteration in rand_ic_iterations:
    spread_2.append(iteration['node_count'][1])


percent_lt_spread = lt_iterations[-1]["node_count"][1]/len(g.nodes())
percent_ic_spread = (ic_iterations[-1]["node_count"][2]+ic_iterations[-1]["node_count"][1])/len(g.nodes())

exec_data = pd.read_csv(f)

test_trial=exec_data.shape[0]
exec_data.loc[test_trial-1, "percent_LT_RIMR"] = percent_lt_spread
exec_data.loc[test_trial-1, "percent_IC_RIMR"] = percent_ic_spread
exec_data.loc[test_trial-1, "percent_LT_RAND"] = rand_lt_iterations[-1]["node_count"][1]/len(g.nodes())
exec_data.loc[test_trial-1, "percent_IC_RAND"] = (rand_ic_iterations[-1]["node_count"][2]+rand_ic_iterations[-1]["node_count"][1])/len(g.nodes())
exec_data.loc[test_trial-1,"LT_threshold"] = lt_threshold
exec_data.loc[test_trial-1,"IC_threshold"] = ic_threshold
exec_data.loc[test_trial-1,"dir"]=direct
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

curip_lt_seeds = pd.read_csv("../../RIM_res/curip_wiki_LT.csv")
curip_ic_seeds = pd.read_csv("../../RIM_res/curip_wiki_IC.csv")

curip_lt_seeds = curip_lt_seeds.loc[:,"Seed_Set"].to_numpy()
curip_ic_seeds = curip_ic_seeds.loc[:,"Seed_Set"].to_numpy()

curip_lt_model = linear_threshold(graph=g, threshold=lt_threshold, seed_set=curip_lt_seeds)
lt_iterations = curip_lt_model.iteration_bunch(lt_num_steps)
print("Final Spread curip, LT",lt_iterations[-1]["node_count"])
# Run the model
curip_ic_model = independent_cascade(graph=g, threshold=ic_threshold, seed_set=curip_ic_seeds)
ic_iterations = curip_ic_model.iteration_bunch(ic_num_steps)
spread_3 = []
for iteration in ic_iterations:
    spread_3.append(iteration['node_count'][1])
print("Final Spread, curip RIM, susceptible, infected and the recovered nodes ",ic_iterations[-1]["node_count"])

curip_lt_set = set(curip_lt_seeds)
seed_set = set(seeds)
print("Intersection LT:",curip_lt_set.intersection(seed_set))
print("Intersection IC:",set(curip_ic_seeds).intersection(seed_set))
len_lt = len(set(curip_lt_seeds).intersection(seed_set))
len_ic = len(set(curip_ic_seeds).intersection(seed_set))


percent_curip_ic_spread = (ic_iterations[-1]["node_count"][2]+ic_iterations[-1]["node_count"][1])/len(g.nodes())
percent_curip_lt_spread=lt_iterations[-1]["node_count"][1]/len(g.nodes())
print("Percent Spread, curip LT, IC",percent_curip_lt_spread, percent_curip_ic_spread)


exec_data = pd.read_csv(f)
test_trial=exec_data.shape[0]
exec_data.loc[test_trial-1, "percent_LT_CU"] = percent_curip_lt_spread
exec_data.loc[test_trial-1, "percent_IC_CU"] = percent_curip_ic_spread
exec_data.loc[test_trial-1, "percent_LT_over"] = len_lt/k
exec_data.loc[test_trial-1, "percent_IC_over"] = len_ic/k

time_RIM = exec_data.loc[test_trial-1, "time(ms)"]

time_IMM_IC = exec_data.loc[test_trial-1, "time_cu_ic"]

time_IMM_LT = exec_data.loc[test_trial-1, "time_cu_lt"]

speedup_LT = time_IMM_LT/time_RIM
speedup_IC = time_IMM_IC/time_RIM

exec_data.loc[test_trial-1, "speedup_LT"] = speedup_LT
exec_data.loc[test_trial-1, "speedup_IC"] = speedup_IC
exec_data.to_csv(f,index=False)