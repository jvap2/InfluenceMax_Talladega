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
hep_K = np.unique(hep_df.loc[:,"K"].to_numpy())
hep_lt_speedup = hep_df.loc[:,"speedup_LT"].to_numpy()
hep_ic_speedup = hep_df.loc[:,"speedup_IC"].to_numpy()
hep_rimr_dist = hep_df.loc[:,"rimr_dist"].to_numpy()
hep_ic_dist = hep_df.loc[:,"ic_dist"].to_numpy()
hep_lt_dist = hep_df.loc[:,"lt_dist"].to_numpy()

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
arxiv_K = np.unique(arxiv_df.loc[:,"K"].to_numpy())
arxiv_lt_speedup = arxiv_df.loc[:,"speedup_LT"].to_numpy()
arxiv_ic_speedup = arxiv_df.loc[:,"speedup_IC"].to_numpy()
arxiv_rimr_dist = arxiv_df.loc[:,"rimr_dist"].to_numpy()
arxiv_ic_dist = arxiv_df.loc[:,"ic_dist"].to_numpy()
arxiv_lt_dist = arxiv_df.loc[:,"lt_dist"].to_numpy()

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
syn_K = np.unique(nd_df.loc[:,"K"].to_numpy())
syn_lt_speedup = nd_df.loc[:,"speedup_LT"].to_numpy()
syn_ic_speedup = nd_df.loc[:,"speedup_IC"].to_numpy()
syn_rimr_dist = nd_df.loc[:,"rimr_dist"].to_numpy()
syn_ic_dist = nd_df.loc[:,"ic_dist"].to_numpy()
syn_lt_dist = nd_df.loc[:,"lt_dist"].to_numpy()


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
amazon_K = np.unique(amazon_df.loc[:,"K"].to_numpy())
amazon_lt_speedup = amazon_df.loc[:,"speedup_LT"].to_numpy()
amazon_ic_speedup = amazon_df.loc[:,"speedup_IC"].to_numpy()
amazon_rimr_dist = amazon_df.loc[:,"rimr_dist"].to_numpy()
amazon_ic_dist = amazon_df.loc[:,"ic_dist"].to_numpy()
amazon_lt_dist = amazon_df.loc[:,"lt_dist"].to_numpy()

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
berk_K = np.unique(berk_df.loc[:,"K"].to_numpy())
berk_lt_speedup = berk_df.loc[:,"speedup_LT"].to_numpy()
berk_ic_speedup = berk_df.loc[:,"speedup_IC"].to_numpy()
berk_rimr_dist = berk_df.loc[:,"rimr_dist"].to_numpy()
berk_ic_dist = berk_df.loc[:,"ic_dist"].to_numpy()
berk_lt_dist = berk_df.loc[:,"lt_dist"].to_numpy()




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
epinions_K = np.unique(epinions_df.loc[:,"K"].to_numpy())
epinions_lt_speedup = epinions_df.loc[:,"speedup_LT"].to_numpy()
epinions_ic_speedup = epinions_df.loc[:,"speedup_IC"].to_numpy()
epinions_rimr_dist = epinions_df.loc[:,"rimr_dist"].to_numpy()
epinions_ic_dist = epinions_df.loc[:,"ic_dist"].to_numpy()
epinions_lt_dist = epinions_df.loc[:,"lt_dist"].to_numpy()


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
wiki_talk_K = np.unique(wiki_talk_df.loc[:,"K"].to_numpy())
wiki_talk_lt_speedup = wiki_talk_df.loc[:,"speedup_LT"].to_numpy()
wiki_talk_ic_speedup = wiki_talk_df.loc[:,"speedup_IC"].to_numpy()
wiki_talk_rimr_dist = wiki_talk_df.loc[:,"rimr_dist"].to_numpy()
wiki_talk_ic_dist = wiki_talk_df.loc[:,"ic_dist"].to_numpy()
wiki_talk_lt_dist = wiki_talk_df.loc[:,"lt_dist"].to_numpy()





fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(14)
fig.suptitle("Epinions Results")


ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=4)
ax2 = plt.subplot2grid((1, 15), (0, 5), colspan=4)
ax3 = plt.subplot2grid((1, 15), (0, 10), colspan=4)


# Add vertical space between the figures
# plt.subplots_adjust(hspace=0.5)


ax1.set_xlabel("K")
ax1.set_ylabel("Spread")
ax1.set_title("LT Spread vs K, HepTh")
ax1.plot(epinions_K, epinions_df_lt_cu, label="CU")
ax1.plot(epinions_K, epinions_df_lt, label="RIMR")
ax1.legend()


ax2.set_xlabel("K")
ax2.set_ylabel("Spread")
ax2.set_title("IC Spread vs K")
ax2.plot(epinions_K, epinions_df_ic_cu, label="CU")
ax2.plot(epinions_K, epinions_df_ic, label="RIMR")
ax2.legend()


ax3.set_xlabel("K")
ax3.set_ylabel("Spread")
ax3.set_title("SpeedUp vs K")
ax3.plot(epinions_K, epinions_lt_speedup, label="LT")
ax3.plot(epinions_K, epinions_ic_speedup, label="IC")
ax3.legend()


fig.savefig("epinions.png")

fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(14)
fig.suptitle("Amazon Results")


ax1 = plt.subplot2grid((1, 15), (0, 0), colspan=4)
ax2 = plt.subplot2grid((1, 15), (0, 5), colspan=4)
ax3 = plt.subplot2grid((1, 15), (0, 10), colspan=4)

ax1.set_xlabel("K")
ax1.set_ylabel("Spread")
ax1.set_title("LT Spread vs K")
ax1.plot(amazon_K, amazon_df_lt_cu, label="CU")
ax1.plot(amazon_K, amazon_df_lt, label="RIMR")
ax1.legend()


ax2.set_xlabel("K")
ax2.set_ylabel("Spread")
ax2.set_title("IC Spread vs K")
ax2.plot(amazon_K, amazon_df_ic_cu, label="CU")
ax2.plot(amazon_K, amazon_df_ic, label="RIMR")
ax2.legend()


ax3.set_xlabel("K")
ax3.set_ylabel("Spread")
ax3.set_title("SpeedUp vs K")
ax3.plot(amazon_K, amazon_lt_speedup, label="LT")
ax3.plot(amazon_K, amazon_ic_speedup, label="IC")
ax3.legend()

fig.savefig("amazon.png")





