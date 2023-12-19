import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import sys


version = int(sys.argv[1]) 

if version == 1:
    file_name = "/meas.csv"
elif version == 2:
    file_name = "/meas_2.csv"
elif version == 3:
    file_name = "/meas_3.csv"

arxiv = "arvix"+file_name
syn = "syn"+file_name
HepTh = "HepTh"+file_name

hep_df = pd.read_csv(HepTh)
arxiv_df = pd.read_csv(arxiv)
syn_df = pd.read_csv(syn)

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

syn_df_ic_cu = syn_df.loc[:,"percent_IC_CU"].to_numpy()
syn_df_lt_cu = syn_df.loc[:,"percent_LT_CU"].to_numpy()
syn_df_ic = syn_df.loc[:,"percent_IC_RIMR"].to_numpy()
syn_df_lt = syn_df.loc[:,"percent_LT_RIMR"].to_numpy()
syn_df_ep = syn_df.loc[:,"eps"].to_numpy()
syn_df_time_lt_cu = syn_df.loc[:,"time_cu_lt"].to_numpy()
syn_df_time_ic_cu = syn_df.loc[:,"time_cu_ic"].to_numpy()
syn_df_time_lt = syn_df.loc[:,"time(ms)"].to_numpy()
syn_df_rimr_epochs = np.unique(syn_df.loc[:,"epochs"].to_numpy())
syn_df_rimr_streams = np.unique(syn_df.loc[:,"streams"].to_numpy())


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
