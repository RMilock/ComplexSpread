def pois_pos_degrees(N, D):
  import numpy as np
  np.random.seed(0)

  'Draw N degrees from a Poissonian sequence with lambda = D and length L'
  def remove_zeros(array):
    #print("array", array, "degarray", np.sum(array))
    its = 0
    while True:
      its += 1
      mask = np.where(array == 0)
      if not mask[0].size: 
        #print(f"Replacing non-0 degrees in {its} iterations")
        return array
      
      'the sum of the degrees must be even'
      psum = np.sum(array)
      #print("psum", psum)
      if not psum % 2: #psum is even return even cover
        while True:
          its += 1
          cover = np.random.poisson(lam = D, size = len(array[mask]))
          #print("even cover?", cover)
          if not np.sum(cover) % 2: break
      else:
        while True: #psum is odd return odd cover
          its += 1
          cover = np.random.poisson(lam = D, size = len(array[mask]))
          if np.sum(cover) % 2: break
      #print("cover final", cover)
      array[mask] = cover

  pos_degrees = np.random.poisson(lam = D, size = N)
  pos_degrees = remove_zeros(pos_degrees)
  #print(pos_degrees)
  return pos_degrees

def dic_nodes_degrees(N, degrees):
    import numpy as np
    np.random.seed(1)
    #print("degrees", degrees, sum(degrees))
    nodes = np.arange(N)
    dic_nodes = nodes.copy()
    np.random.shuffle(dic_nodes)
    dic_nodes = {k:v for k in dic_nodes for v in np.sort(degrees)[np.where(dic_nodes == k)]}
    sorted_nodes = np.array([x for x in dic_nodes.keys()])
    #print(f'nodes: {nodes}, sorted_nodes: {sorted_nodes}', "dic_nodes", dic_nodes, np.sort(degrees))
    return dic_nodes

def rhu(n, decimals=0, integer = False): #round_half_up
    import math
    multiplier = 10 ** decimals
    res = math.floor(n*multiplier + 0.5) / multiplier
    if integer: return int(res)
    return res

def NNOverl_pois_net(N, ext_D, p, add_edges_only = True, conn_flag = False):
    #Nearest Neighbors Overlapping
    from itertools import chain
    from definitions import replace_edges_from, check_loops_parallel_edges,\
        remove_loops_parallel_edges, N_D_std_D, infos_sorted_nodes, connect_net
    import datetime as dt
    import numpy as np
    import random
    import networkx as nx


    def edges_node(x):
        return [(i,j) for i,j in all_edges if i == x]

    D = ext_D
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    degrees = pois_pos_degrees(N,D)
    dic_nodes = dic_nodes_degrees(N, degrees)

    deg_mean = np.mean(degrees)   
    
    'rewire left and right for the max even degree'
    dsc_sorted_nodes = dic_nodes_degrees(N, degrees)
    print(f'dsc_sorted_nodes: {dsc_sorted_nodes}',)
    
    #dsc_sorted_nodes = {k: v for k,v in sorted( G.degree(), key = lambda x: x[1], reverse=True)}
    edges = set()
    for node in dsc_sorted_nodes.keys():
        k = dsc_sorted_nodes[node]//2
        for i in range(1, k + 1):
            edges.add((node, (node+i)%N))
            edges.add((node, (node-i)%N))
        if dsc_sorted_nodes[node] % 2 == 1: 
            edges.add((node, (node+k+1)%N))
        elif dsc_sorted_nodes[node] % 2 != 0: 
            print("Error of Wiring: dsc[node]%2", dsc_sorted_nodes[node] % 2); break
    G.add_edges_from(edges)
    remove_loops_parallel_edges(G)

    _, D, _ = N_D_std_D(G)
    if D != deg_mean: print("!! avg_deg_Overl_Rew - avg_deg_Conf_Model = ", D - deg_mean)

    if add_edges_only and p != 0: #add on top of the distr, a new long-range edge
        long_range_edge_add(G, p = p)
    
    adeg_OR = sum([j for i,j in G.degree()])/G.number_of_nodes()
    print("Rel Error wrt to ext_D %s %%" % (rhu((adeg_OR - ext_D)/ext_D,1 )*100))
    print("Rel Error wrt to deg_mean %s %%" % (rhu( (adeg_OR - deg_mean)/deg_mean,1 )*100))

    '''
    else: #remove local edge and add long-range one
        start_time = dt.datetime.now()
        all_edges = [list(G.edges(node)) for node in pos_deg_nodes(G)]
        all_edges = list(chain.from_iterable(all_edges))
        initial_length = len(all_edges)
        for node in pos_deg_nodes(G):
            left_nodes = list(pos_deg_nodes(G))
            left_nodes.remove(node) 
            #print("edges of the node %s: %s" % (node, edges_node(node)) ) 
            i_rmv, j_rmv = random.choice(edges_node(node))  
            if random.random() < p and len(edges_node(j_rmv)) > 1: #if, with prob p, the "node" is not the only friend of "j_rmv" 
                all_edges.remove((i_rmv,j_rmv)); all_edges.remove((j_rmv,i_rmv))
                re_link = random.choice(left_nodes)
                #print("rmv_choice: (%s,%s)" % (i_rmv, j_rmv), "relink with node: ", re_link)
                all_edges.append((node, re_link)); all_edges.append((re_link, node))
                all_edges = [(node,j) for node in range(len(all_edges)) for j in [j for i,j in all_edges if i == node] ]
                #print("all_edges", all_edges)
        print("len(all_edges)_final", len(all_edges), "is? equal to start", initial_length )
        replace_edges_from(G, all_edges)
        remove_loops_parallel_edges(G, False)
        print(f"Time for add_edges = {add_edges_only}:", dt.datetime.now()-start_time)
    '''

    check_loops_parallel_edges(G)
    infos_sorted_nodes(G, num_sorted_nodes=False)
    connect_net(G, conn_flag = conn_flag)

    _,D,_ = N_D_std_D(G)
    print(f"End of wiring with average degree {D} vs {ext_D}")
    print(f'G.is_connected(): {nx.is_connected(G)}',)

    print(f'G.edges(): {G.edges()}',)

    return G, dic_nodes

import matplotlib.pylab as plt
import networkx as nx

G, dic_nodes = NNOverl_pois_net(10,2,p = 0)

#plt.figure(figsize = (8,5))
#nx.draw_circular(G, with_labels = True, width = 1, node_size = 20)
#plt.show()

fig, ax = plt.subplots(figsize = (14,10))
labels = {k:f"Degre[{k}]={v}" for k,v in dic_nodes.items()}

#print(f'labels: {labels}',)
nx.draw_circular(G, ax = ax, with_labels = True, labels = labels, width = 1, node_size = 1e3, alpha = 0.9, node_color = "orange", font_size = 10)
plt.show()

