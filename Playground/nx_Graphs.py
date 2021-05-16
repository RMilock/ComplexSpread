import networkx as nx
import matplotlib.pyplot as plt

string = "G = nx.binomial_tree(10)"
name = string[4:]

exec(string)

nx.draw_circular(G, with_labels = False)
file_path = "".join(("./Graph_types/", name, ".png"))
print(file_path)
plt.savefig(file_path)
plt.show()