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
# plt.plot(nd_blocks, nd_time, label='ND')
# plt.plot(amazon_blocks, amazon_time, label='Amazon')
# plt.xlabel('Number of blocks')
# plt.ylabel('Time (s)')
# plt.title('Time to scale')
# plt.legend()


# plt.savefig('scale.png')

## Plot speedup relative to 1 block
speedup_nd = nd_time[0] / nd_time
speedup_amazon = amazon_time[0] / amazon_time

plt.plot(nd_blocks, speedup_nd, label='ND')
plt.plot(amazon_blocks, speedup_amazon, label='Amazon')
plt.xlabel('Number of blocks')
plt.ylabel('Speedup')
plt.title('Speedup relative to 1 block')
plt.legend()

plt.savefig('speedup.png')
