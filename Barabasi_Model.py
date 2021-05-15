import networkx as nx
import random
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from itertools import product
from definitions import save_log_params, plot_save_nes

def bam(N,m,N0):
    '''    
    Arguments:
    1) N: number of nodes in the graph   
    2) m: number of links to be added at each time
    3) N0: starting fully connected clique
    '''
    
    'Creates an empty graph'
    G = nx.Graph()
    
    'adds the N0 initial nodes'
    G.add_nodes_from(range(N0))
    edges = []
    
    'creates the initial clique connecting all the N0 nodes'
    edges = [(i,j) for i in range(N0) for j in range(i,N0) if i!=j]
    
    'adds the initial clique to the network'
    G.add_edges_from(edges)

    'list to store the nodes to be selected for the preferential attachment.'
    'instead of calculating the probability of being selected a trick is used: if a node has degree k, it will appear'
    'k times in the list. This is equivalent to select them according to their probability.'
    prob = []
    
    'runs over all the reamining nodes'
    for i in range(N0,N):
        G.add_node(i)
        'for each new node, creates m new links'
        for j in range(m):
            'creates the list of nodes'
            for k in list(G.nodes):
                'add to prob a node as many time as its degree'
                for _ in range(G.degree(k)):
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
k_prog = np.arange(2,18,2) # these are the fully connected initial cliques
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
beta_prog = np.linspace(0.001,1,15)
mu_prog = np.linspace(0.01,1,13)
p_prog = [0]
R0_min = 0; R0_max = 3    
folder = "B-A_Model"


'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  'since D_real ~ 2*D (D here is fixing only the m and N0), R0_max-folder ~ 2*R0_max'
  if R0_min < beta*D/mu < R0_max: 
    m, N0 = D,D
    done_iterations+=1
    print("\nIterations left: %s" % ( total_iterations - done_iterations ) )
    text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s; \nTotal Iterations: %s;\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max, total_iterations)

    save_log_params(folder = folder, text = text, done_iterations = done_iterations)

    plot_save_nes(G = bam(N, m = m, N0 = N0), m = m, N0 = N0,
    p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
    plot_save_nes(G = bam(N, m = m, N0 = N0), m = m, N0 = N0,
    p = p, folder = folder, adj_or_sir="SIR", beta = beta, mu = mu, done_iterations=done_iterations)