import pandas as pd


mapping = pd.read_csv("../../Graph_Data_Storage/arvix_mapping.csv")
seeds = pd.read_csv("../../RIM_res/res_arvix.csv")
cu_rip_seed_lt = pd.read_csv("../../RIM_res/curip_arvix_LT.csv")
cu_rip_seed_ic = pd.read_csv("../../RIM_res/curip_arvix_IC.csv")

mapping = mapping.loc[:,"old_id"].to_numpy()    
seeds = seeds.loc[:,"Seed_Set"].to_numpy()
cu_rip_seed_lt= cu_rip_seed_lt.loc[:,"Seed_Set"].to_numpy()
cu_rip_seed_ic= cu_rip_seed_ic.loc[:,"Seed_Set"].to_numpy()
new_seeds = []
for seed in seeds:
    new_seeds.append(mapping[seed])

lt_seeds = []
for seed in cu_rip_seed_lt:
    lt_seeds.append(mapping[seed])

ic_seeds = []
for seed in cu_rip_seed_ic:
    ic_seeds.append(mapping[seed])



new_df = pd.DataFrame(list(zip(new_seeds)), columns=["Seed_Set"])
new_df.to_csv("../../RIM_res/res_arvix_new.csv", index=False)

new_lt_df = pd.DataFrame(list(zip(lt_seeds)), columns=["Seed_Set"])
new_lt_df.to_csv("../../RIM_res/curip_arvix_LT_new.csv", index=False)

new_ic_df = pd.DataFrame(list(zip(ic_seeds)), columns=["Seed_Set"])
new_ic_df.to_csv("../../RIM_res/curip_arvix_IC_new.csv", index=False)