import os
import sys
import random
import numpy as np

# command parameters: # <num_persons> <num_servers> <seed>
#
# 2000000 500000 4
#

UsesOut  = open("../Graph_Data_Storage/hetero.csv", "w")

line2 = "person,server\n"
UsesOut.write(line2)

num_persons = int(sys.argv[1])
print(num_persons)
num_servers = int(sys.argv[2])
print(num_servers)
print(int(sys.argv[3]))
np.random.seed(int(sys.argv[3]))

########### PRINT SOCIAL RECORDS ##########

for person in range(0, num_persons):
  num_used = int( np.random.exponential(20.0) )
  if num_used > 20: num_used = int( num_used / 10 )     # let's not generate too many edges

  for i in range(0, num_used):
    server = int( np.random.random() * num_servers )
    UsesOut.write(str(person) + "," + str(server) + "\n")