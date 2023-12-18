import pandas as pd


mapping = pd.read_csv("../../Graph_Data_Storage/wiki_mapping.csv")
seeds = pd.read_csv("../../RIM_res/res_wiki.csv")
cu_rip_seed_lt = pd.read_csv("../../RIM_res/curip_wiki_LT.csv")
cu_rip_seed_ic = pd.read_csv("../../RIM_res/curip_wiki_IC.csv")

mapping = mapping.loc[:,"old_id"].to_numpy()    
seeds = seeds.loc[:,"Seed_Set"].to_numpy()
cu_rip_seed_lt = cu_rip_seed_lt.loc[:,"Seed_Set"].to_numpy()
cu_rip_seed_ic = cu_rip_seed_ic.loc[:,"Seed_Set"].to_numpy()
new_seeds = []
for seed in seeds:
    new_seeds.append(mapping[seed])

lt_seeds = []
for seed in cu_rip_seed_lt:
    print(seed)
    lt_seeds.append(mapping[seed])

ic_seeds = []
for seed in cu_rip_seed_ic:
    ic_seeds.append(mapping[seed])



new_df = pd.DataFrame(list(zip(new_seeds)), columns=["Seed_Set"])
new_df.to_csv("../../RIM_res/res_wiki_new.csv", index=False)

new_lt_df = pd.DataFrame(list(zip(lt_seeds)), columns=["Seed_Set"])
new_lt_df.to_csv("../../RIM_res/curip_wiki_LT_new.csv", index=False)

new_ic_df = pd.DataFrame(list(zip(ic_seeds)), columns=["Seed_Set"])
new_ic_df.to_csv("../../RIM_res/curip_wiki_IC_new.csv", index=False)
