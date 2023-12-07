import pandas as pd


mapping = pd.read_csv("../../Graph_Data_Storage/arvix_mapping.csv")
seeds = pd.read_csv("../../RIM_res/res_arvix.csv")

mapping = mapping.loc[:,"old_id"].to_numpy()    
seeds = seeds.loc[:,"Seed_Set"].to_numpy()
new_seeds = []
for seed in seeds:
    new_seeds.append(mapping[seed])


new_df = pd.DataFrame(list(zip(new_seeds)), columns=["Seed_Set"])
new_df.to_csv("../../RIM_res/res_arvix_new.csv", index=False)

