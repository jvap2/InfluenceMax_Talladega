import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import sys


# version = int(sys.argv[1]) 

# if version == 1:
#     file_name = "/meas.csv"
# elif version == 2:
#     file_name = "/meas_2.csv"
# elif version == 3:
#     file_name = "/meas_3.csv"

file_name = "/meas_8.csv"

arxiv = "arvix"+file_name
nd = "nd"+file_name
berk = "berk"+file_name
epinions = "epinions"+file_name
wiki_talk = "wiki_talk"+file_name
amazon = "amazon"+file_name
HepTh = "HepTh"+file_name

hep_df = pd.read_csv(HepTh)
arxiv_df = pd.read_csv(arxiv)
amazon_df = pd.read_csv(amazon)
nd_df = pd.read_csv(nd)
berk_df = pd.read_csv(berk)
epinions_df = pd.read_csv(epinions)
wiki_talk_df = pd.read_csv(wiki_talk)


hep_df_ic_cu = hep_df.loc[:,"percent_IC_CU"].to_numpy()
hep_df_lt_cu = hep_df.loc[:,"percent_LT_CU"].to_numpy()
hep_df_ic = hep_df.loc[:,"percent_IC_RIMR"].to_numpy()
hep_df_lt = hep_df.loc[:,"percent_LT_RIMR"].to_numpy()
hep_df_ep = hep_df.loc[:,"eps"].to_numpy()
hep_df_time_lt_cu = hep_df.loc[:,"time_cu_lt"].to_numpy()
hep_df_time_ic_cu = hep_df.loc[:,"time_cu_ic"].to_numpy()
hep_df_time_lt = hep_df.loc[:,"time(ms)"].to_numpy()
hep_df_rimr_epochs = np.unique(hep_df.loc[:,"epochs"].to_numpy())
hep_df_rimr_streams = np.unique(hep_df.loc[:,"streams"].to_numpy())

arxiv_df_ic_cu = arxiv_df.loc[:,"percent_IC_CU"].to_numpy()
arxiv_df_lt_cu = arxiv_df.loc[:,"percent_LT_CU"].to_numpy()
arxiv_df_ic = arxiv_df.loc[:,"percent_IC_RIMR"].to_numpy()
arxiv_df_lt = arxiv_df.loc[:,"percent_LT_RIMR"].to_numpy()
arxiv_df_ep = arxiv_df.loc[:,"eps"].to_numpy()
arxiv_df_time_lt_cu = arxiv_df.loc[:,"time_cu_lt"].to_numpy()
arxiv_df_time_ic_cu = arxiv_df.loc[:,"time_cu_ic"].to_numpy()
arxiv_df_time_lt = arxiv_df.loc[:,"time(ms)"].to_numpy()
arxiv_df_rimr_epochs = np.unique(arxiv_df.loc[:,"epochs"].to_numpy())
arxiv_df_rimr_streams = np.unique(arxiv_df.loc[:,"streams"].to_numpy())

syn_df_ic_cu = nd_df.loc[:,"percent_IC_CU"].to_numpy()  
syn_df_lt_cu = nd_df.loc[:,"percent_LT_CU"].to_numpy()
syn_df_ic = nd_df.loc[:,"percent_IC_RIMR"].to_numpy()
syn_df_lt = nd_df.loc[:,"percent_LT_RIMR"].to_numpy()
syn_df_ep = nd_df.loc[:,"eps"].to_numpy()
syn_df_time_lt_cu = nd_df.loc[:,"time_cu_lt"].to_numpy()
syn_df_time_ic_cu = nd_df.loc[:,"time_cu_ic"].to_numpy()
syn_df_time_lt = nd_df.loc[:,"time(ms)"].to_numpy()
syn_df_rimr_epochs = np.unique(nd_df.loc[:,"epochs"].to_numpy())
syn_df_rimr_streams = np.unique(nd_df.loc[:,"streams"].to_numpy())  


amazon_df_ic_cu = amazon_df.loc[:,"percent_IC_CU"].to_numpy()
amazon_df_lt_cu = amazon_df.loc[:,"percent_LT_CU"].to_numpy()
amazon_df_ic = amazon_df.loc[:,"percent_IC_RIMR"].to_numpy()
amazon_df_lt = amazon_df.loc[:,"percent_LT_RIMR"].to_numpy()
amazon_df_ep = amazon_df.loc[:,"eps"].to_numpy()
amazon_df_time_lt_cu = amazon_df.loc[:,"time_cu_lt"].to_numpy()
amazon_df_time_ic_cu = amazon_df.loc[:,"time_cu_ic"].to_numpy()
amazon_df_time_lt = amazon_df.loc[:,"time(ms)"].to_numpy()
amazon_df_rimr_epochs = np.unique(amazon_df.loc[:,"epochs"].to_numpy())
amazon_df_rimr_streams = np.unique(amazon_df.loc[:,"streams"].to_numpy())

berk_df_ic_cu = berk_df.loc[:,"percent_IC_CU"].to_numpy()
berk_df_lt_cu = berk_df.loc[:,"percent_LT_CU"].to_numpy()
berk_df_ic = berk_df.loc[:,"percent_IC_RIMR"].to_numpy()
berk_df_lt = berk_df.loc[:,"percent_LT_RIMR"].to_numpy()
berk_df_ep = berk_df.loc[:,"eps"].to_numpy()
berk_df_time_lt_cu = berk_df.loc[:,"time_cu_lt"].to_numpy()
berk_df_time_ic_cu = berk_df.loc[:,"time_cu_ic"].to_numpy()
berk_df_time_lt = berk_df.loc[:,"time(ms)"].to_numpy()
berk_df_rimr_epochs = np.unique(berk_df.loc[:,"epochs"].to_numpy())
berk_df_rimr_streams = np.unique(berk_df.loc[:,"streams"].to_numpy())

epinions_df_ic_cu = epinions_df.loc[:,"percent_IC_CU"].to_numpy()
epinions_df_lt_cu = epinions_df.loc[:,"percent_LT_CU"].to_numpy()
epinions_df_ic = epinions_df.loc[:,"percent_IC_RIMR"].to_numpy()
epinions_df_lt = epinions_df.loc[:,"percent_LT_RIMR"].to_numpy()
epinions_df_ep = epinions_df.loc[:,"eps"].to_numpy()
epinions_df_time_lt_cu = epinions_df.loc[:,"time_cu_lt"].to_numpy()
epinions_df_time_ic_cu = epinions_df.loc[:,"time_cu_ic"].to_numpy()
epinions_df_time_lt = epinions_df.loc[:,"time(ms)"].to_numpy()
epinions_df_rimr_epochs = np.unique(epinions_df.loc[:,"epochs"].to_numpy())
epinions_df_rimr_streams = np.unique(epinions_df.loc[:,"streams"].to_numpy())


wiki_talk_df_ic_cu = wiki_talk_df.loc[:,"percent_IC_CU"].to_numpy()
wiki_talk_df_lt_cu = wiki_talk_df.loc[:,"percent_LT_CU"].to_numpy()
wiki_talk_df_ic = wiki_talk_df.loc[:,"percent_IC_RIMR"].to_numpy()
wiki_talk_df_lt = wiki_talk_df.loc[:,"percent_LT_RIMR"].to_numpy()
wiki_talk_df_ep = wiki_talk_df.loc[:,"eps"].to_numpy()
wiki_talk_df_time_lt_cu = wiki_talk_df.loc[:,"time_cu_lt"].to_numpy()
wiki_talk_df_time_ic_cu = wiki_talk_df.loc[:,"time_cu_ic"].to_numpy()
wiki_talk_df_time_lt = wiki_talk_df.loc[:,"time(ms)"].to_numpy()
wiki_talk_df_rimr_epochs = np.unique(wiki_talk_df.loc[:,"epochs"].to_numpy())
wiki_talk_df_rimr_streams = np.unique(wiki_talk_df.loc[:,"streams"].to_numpy())




fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Spread")
ax1.set_title("Spread vs Epsilon (IC)")
ax1.set_ylim(0, 1)
ax1.set_xlim(.1,.9)
ax1.plot(hep_df_ep, hep_df_ic_cu, label="IC_CU")
ax1.plot(hep_df_ep, hep_df_ic, label="IC_RIMR")
ax1.legend()

ax2.set_xlabel("Epsilon")
ax2.set_ylabel("Spread")
ax2.set_title("Spread vs Epsilon (LT)")
ax2.set_ylim(0, 1)
ax2.set_xlim(.1,.9)
ax2.plot(hep_df_ep, hep_df_lt_cu, label="LT_CU")
ax2.plot(hep_df_ep, hep_df_lt, label="LT_RIMR")
ax2.legend()

fig.savefig("hep_spread_vs_eps.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Spread")
ax1.set_title("Spread vs Epsilon (IC)")
ax1.set_ylim(0, 1)
ax1.set_xlim(.1,.9)
ax1.plot(arxiv_df_ep, arxiv_df_ic_cu, label="IC_CU")
ax1.plot(arxiv_df_ep, arxiv_df_ic, label="IC_RIMR")
ax1.legend()

ax2.set_xlabel("Epsilon")
ax2.set_ylabel("Spread")
ax2.set_title("Spread vs Epsilon (LT)")
ax2.set_ylim(0, 1)
ax2.set_xlim(.1,.9)
ax2.plot(arxiv_df_ep, arxiv_df_lt_cu, label="LT_CU")
ax2.plot(arxiv_df_ep, arxiv_df_lt, label="LT_RIMR")
ax2.legend()

fig.savefig("arxiv_spread_vs_eps.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Spread")
ax1.set_title("Spread vs Epsilon (IC)")
ax1.set_ylim(0, 1)
ax1.set_xlim(.1,.9)
ax1.plot(syn_df_ep, syn_df_ic_cu, label="IC_CU")
ax1.plot(syn_df_ep, syn_df_ic, label="IC_RIMR")
ax1.legend()

ax2.set_xlabel("Epsilon")
ax2.set_ylabel("Spread")
ax2.set_title("Spread vs Epsilon (LT)")
ax2.set_ylim(0, 1)
ax2.set_xlim(.1,.9)
ax2.plot(syn_df_ep, syn_df_lt_cu, label="LT_CU")
ax2.plot(syn_df_ep, syn_df_lt, label="LT_RIMR")
ax2.legend()

fig.savefig("syn_spread_vs_eps.png")


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Time (ms)")
ax1.set_title("Time vs Epsilon")
ax1.plot(hep_df_ep, hep_df_time_lt_cu, label="LT_CU")
ax1.plot(hep_df_ep, hep_df_time_ic_cu, label="IC_CU")
ax1.plot(hep_df_ep, hep_df_time_lt, label="RIMR")
ax1.legend()
fig.savefig("hep_time_vs_eps.png")

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Time (ms)")
ax1.set_title("Time vs Epsilon")
ax1.plot(arxiv_df_ep, arxiv_df_time_lt_cu, label="LT_CU")
ax1.plot(arxiv_df_ep, arxiv_df_time_ic_cu, label="IC_CU")
ax1.plot(arxiv_df_ep, arxiv_df_time_lt, label="RIMR")
ax1.legend()
fig.savefig("arxiv_time_vs_eps.png")

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Time (ms)")
ax1.set_title("Time vs Epsilon")
ax1.plot(syn_df_ep, syn_df_time_lt_cu, label="LT_CU")
ax1.plot(syn_df_ep, syn_df_time_ic_cu, label="IC_CU")
ax1.plot(syn_df_ep, syn_df_time_lt, label="RIMR")
ax1.legend()
fig.savefig("syn_time_vs_eps.png")

hep_lt_speedup = np.divide(hep_df_time_lt_cu, hep_df_time_lt)
hep_ic_speedup = np.divide(hep_df_time_ic_cu, hep_df_time_lt)
arxiv_lt_speedup = np.divide(arxiv_df_time_lt_cu, arxiv_df_time_lt)
arxiv_ic_speedup = np.divide(arxiv_df_time_ic_cu, arxiv_df_time_lt)
syn_lt_speedup = np.divide(syn_df_time_lt_cu, syn_df_time_lt)
syn_ic_speedup = np.divide(syn_df_time_ic_cu, syn_df_time_lt)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Speedup")
ax1.set_title("Speedup vs Epsilon")
ax1.plot(hep_df_ep, hep_lt_speedup, label="LT_CU")
ax1.plot(hep_df_ep, hep_ic_speedup, label="IC_CU")
ax1.legend()
fig.savefig("hep_speedup_vs_eps.png")

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Speedup")
ax1.set_title("Speedup vs Epsilon")
ax1.plot(arxiv_df_ep, arxiv_lt_speedup, label="LT_CU")
ax1.plot(arxiv_df_ep, arxiv_ic_speedup, label="IC_CU")
ax1.legend()

fig.savefig("arxiv_speedup_vs_eps.png")

fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Speedup")
ax1.set_title("Speedup vs Epsilon")
ax1.plot(syn_df_ep, syn_lt_speedup, label="LT_CU")
ax1.plot(syn_df_ep, syn_ic_speedup, label="IC_CU")
ax1.legend()
fig.savefig("syn_speedup_vs_eps.png")
