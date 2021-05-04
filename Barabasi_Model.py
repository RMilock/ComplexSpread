import networkx as nx
import random
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from itertools import product
from definitions import save_log_params, rhu, plot_save_net, plot_save_sir

def bam(N,m,m0 = 4):
    '''    
    Arguments:
    1) N: number of nodes in the graph   
    2) m: number of links to be added at each time
    '''
    
    'Creates an empty graph'
    G = nx.Graph()
    
    'size of the initial clique of the network'
    m0 = m0
    
    'adds the m0 initial nodes'
    G.add_nodes_from(range(m0))
    edges = []
    
    'creates the initial clique connecting all the m0 nodes'
    for i in range(m0):
        for j in range(i,m0):                
                if i != j: #avoid loops
                    edges.append((i,j))
    
    'adds the initial clique to the network'
    G.add_edges_from(edges)

    'list to store the nodes to be selected for the preferential attachment.'
    'instead of calculating the probability of being selected a trick is used: if a node has degree k, it will appear'
    'k times in the list. This is equivalent to select them according to their probability.'
    prob = []
    
    'runs over all the reamining nodes'
    for i in range(m0,N):
        G.add_node(i)
        'for each new node, creates m new links'
        for j in range(m):
            'creates the list of nodes'
            for k in list(G.nodes):
                'each node is added to the list according to its degree'
                for _ in range(nx.degree(G,k)):
                    prob.append(k)
            'picks up a random node, so nodes will be selected proportionally to their degree'
            node = random.choice(prob)
            
            G.add_edge(node,i)
        
            'the list must be created from 0 for every link since with every new link probabilities change'
            prob.clear()
    'returns the graph'

    return G

N = int(1e3); p_max = 0 

'progression of net-parameters'
k_prog = np.arange(2,18,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.01,1,10)
beta_prog = np.linspace(0.0001,.5,15)
p_prog = [0]
R0_min = 0; R0_max = 3


'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("\nIterations left: %s" % ( total_iterations - done_iterations ) )
    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max)
    
    folder = "Barabasi"
    print("N: %s, D: %s" % (N,D) ) 
    G = nx.barabasi_albert_graph(N,D)
    
    'plot G, degree distribution and the adiaciency matrix and save them'
    save_log_params(folder = folder, text = text)
    if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
      plot_save_net(G = G, folder = folder, D = D, p = p, done_iterations = done_iterations)
      saved_nets.append("N%s_D%s_p%s"% (N,D,rhu(p,3)))
      #print(saved_nets)
    plot_save_sir(G, folder = folder, beta = beta, D = D, mu = mu, p = p_max, done_iterations = done_iterations)