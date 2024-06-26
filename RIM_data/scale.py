import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
nd = pd.read_csv('nd/scale.csv')
amazon = pd.read_csv('amazon/scale.csv')

nd_blocks = nd.loc[:,"blocks"].to_numpy()
nd_time = nd.loc[:,"time(ms)"].to_numpy()

amazon_blocks = amazon.loc[:,"blocks"].to_numpy()
amazon_time = amazon.loc[:,"time(ms)"].to_numpy()

# Plot the data
plt.figure(figsize=(7, 5))  # Set the figure size to 8 inches by 6 inches

plt.plot(nd_blocks, nd_time, label='ND')
plt.plot(amazon_blocks, amazon_time, label='Amazon')
plt.xlabel('Number of blocks')
plt.ylabel('Time (ms)')
plt.title('RIMR Time to scale')
plt.legend()

plt.savefig('scale.png')

## Plot speedup relative to 1 block
# speedup_nd = nd_time[0] / nd_time
# speedup_amazon = amazon_time[0] / amazon_time
# speedup_ideal = np.arange(1, 1+len(nd_blocks))

# plt.plot(nd_blocks, speedup_nd, label='ND')
# plt.plot(amazon_blocks, speedup_amazon, label='Amazon')
# plt.plot(nd_blocks, speedup_ideal, label='Ideal')
# plt.xlabel('Number of blocks')
# plt.ylabel('Speedup')
# plt.title('RIMR Speedup relative to 1 block')
# plt.legend()

# plt.savefig('speedup.png')
