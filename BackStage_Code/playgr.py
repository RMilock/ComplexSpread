import networkx as nx
import numpy as np

# with is like your try .. finally block in this case
dir = "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/ComplexSpread/BackStage_Code/"
txtname = "stats.txt"
file_name = dir + txtname
with open(file_name, 'r') as file:
    # read a list of lines into data
    data = file.readlines()


for x in data:
    if "N" in x and "=" in x:
        print(f'data: {data}',f'x: {x}',)
        print('np.where(data == x)', data.index(x), )
        data[data.index(x)] = "N = int(1e3)\n"
    if "N" in x and "=" in x:
        print(f'data: {data}',f'x: {x}',)
        print('np.where(data == x)', data.index(x), )
        data[data.index(x)] = "N = int(1e3)\n"
    break


print("data",data)
print("Your name: ", data[0])

# now change the 2nd line, note that you have to add a newline
#data[1] = 'Mage\n'

# and write everything back
txtname = "new_"+txtname
file_name = dir+txtname
with open(file_name, 'w') as file:
    file.writelines( data )  
