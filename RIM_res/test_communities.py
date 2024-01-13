import pandas as pd
import numpy as np
import networkx as nx
import sys

vers = int(sys.argv[1])
rim_vers = int(sys.argv[2])
if vers ==2:
    File = "../../Graph_Data_Storage/ca-GrQc.txt"
    top = pd.read_csv("res_arvix_new.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_arvix_LT_new.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_arvix_IC_new.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/arvix/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/arvix/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/arvix/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/arvix/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/arvix/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/arvix/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/arvix/meas_7.csv"
if vers ==3:
    File = "../../Graph_Data_Storage/ca-HepTh.txt"
    top = pd.read_csv("res_hepth.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_hepth_LT_new.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_hepth_IC_new.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/HepTh/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/HepTh/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/HepTh/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/HepTh/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/HepTh/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/HepTh/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/HepTh/meas_7.csv"
if vers ==4:
    File = "../../Graph_Data_Storage/wiki-Vote.txt"
    top = pd.read_csv("res_wiki.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_wiki_LT.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_wiki_IC.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/wiki-vote/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/wiki-vote/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/wiki-vote/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/wiki-vote/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/wiki-vote/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/wiki-vote/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/wiki-vote/meas_7.csv"
if vers ==5:
    File = "../../Graph_Data_Storage/soc-Epinions1.txt"
    top = pd.read_csv("res_ep.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_ep_LT.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_ep_IC.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/epinions/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/epinions/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/epinions/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/epinions/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/epinions/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/epinions/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/epinions/meas_7.csv"
if vers ==6:
    File = "../../Graph_Data_Storage/amazon0302.txt"
    top = pd.read_csv("res_amazon.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_amazon_LT.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_amazon_IC.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/amazon/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/amazon/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/amazon/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/amazon/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/amazon/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/amazon/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/amazon/meas_7.csv"
if vers == 7:
    File = "../Graph_Data_Storage/web-NotreDame.txt"
    top = pd.read_csv("res_nd.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_nd_LT.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_nd_IC.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/nd/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/nd/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/nd/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/nd/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/nd/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/nd/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/nd/meas_7.csv"
if vers ==8:
    File ="../../Graph_Data_Storage/web-Google.txt"
    top = pd.read_csv("res_google.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_google_LT.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_google_IC.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/google/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/google/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/google/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/google/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/google/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/google/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/google/meas_7.csv"
if vers ==9:
    File = "../../Graph_Data_Storage/web-BerkStan.txt"
    top = pd.read_csv("res_berk_new.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_berk_LT_new.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_berk_IC_new.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/berk/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/berk/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/berk/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/berk/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/berk/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/berk/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/berk/meas_7.csv"
if vers ==10:
    File = "../../Graph_Data_Storage/wiki-Talk.txt"
    top = pd.read_csv("res_wiki_talk.csv").to_numpy().flatten()
    lt_set = pd.read_csv("curip_wiki_talk_LT.csv").to_numpy().flatten()
    ic_set = pd.read_csv("curip_wiki_talk_IC.csv").to_numpy().flatten()
    if rim_vers==1:
        time_file = "../RIM_data/wiki_talk/meas.csv"
    elif rim_vers==2:
        time_file = "../RIM_data/wiki_talk/meas_2.csv"
    elif rim_vers==3:
        time_file = "../RIM_data/wiki_talk/meas_3.csv"
    elif rim_vers==4:
        time_file = "../RIM_data/wiki_talk/meas_4.csv"
    elif rim_vers==5:
        time_file = "../RIM_data/wiki_talk/meas_5.csv"
    elif rim_vers==6:
        time_file = "../RIM_data/wiki_talk/meas_6.csv"
    elif rim_vers==7:
        time_file = "../RIM_data/wiki_talk/meas_7.csv"

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
d_ic = average_distance(g, ic_set)
d_lt = average_distance(g, lt_set)
print("RIMR: ",d)
print("CURIP IC: ",d_ic)
print("CURIP LT: ",d_lt)


df = pd.read_csv(time_file)
trial = df.shape[0]
df.loc[trial-1, "rimr_dist"] = d
df.loc[trial-1, "ic_dist"] = d_ic
df.loc[trial-1, "lt_dist"] = d_lt
df.to_csv(time_file, index=False)