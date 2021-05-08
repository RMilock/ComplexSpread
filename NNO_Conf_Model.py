import numpy as np
from itertools import product
from definitions import save_log_params, rhu, plot_save_net, \
    remove_loops_parallel_edges, plot_save_sir, config_pois_model, replace_edges_from, \
    infos_sorted_nodes


def NN_Overlapping_Conf_Model(N, D, p, add_edges_only = False):
    from itertools import chain
    import datetime as dt
    import random

    
    def edges_node(x):
        return [(i,j) for i,j in all_edges if i == x]

    G = config_pois_model(N,D)
    adeg_CM = sum([j for i,j in G.degree()])/G.number_of_nodes()

    dsc_sorted_nodes = {k: v for k,v in sorted( G.degree(), key = lambda x: x[1], reverse=True)}
    edges = set()
    for node in dsc_sorted_nodes.keys():
        k = dsc_sorted_nodes[node]//2
        for i in range(1, k + 1):
            edges.add((node, (node+i)%N))
            edges.add((node, (node-i)%N))
        if dsc_sorted_nodes[node] % 2 == 1: edges.add((node, (node+k+1)%N))
        elif dsc_sorted_nodes[node] % 2 != 0: print("Error of Wiring: dsc[node]%2", dsc_sorted_nodes[node] % 2); break
    replace_edges_from(G, edges)
    #print("bef rmv loops and //", G.edges(), G.degree(), sum([j for i,j in G.degree()])/G.number_of_nodes())

    remove_loops_parallel_edges(G)
    #print("after rmv loops and //", G.edges(), G.degree(), sum([j for i,j in G.degree()])/G.number_of_nodes())
    #print([(i,npr.choice(G.nodes().pop(i))) for i in G.edges() if i == 0])

    all_edges = [list(G.edges(node)) for node in G.nodes()]
    all_edges = list(chain.from_iterable(all_edges))
    initial_length = len(all_edges)
    #print( "inital edges ", all_edges )

    start_time = dt.datetime.now()

    if add_edges_only:
        for node in G.nodes():
            left_nodes = list(G.nodes())
            left_nodes.remove(node) 
            re_link = random.choice( left_nodes )
            if random.random() < p:
                all_edges.append((node,re_link))

    else:
        for node in G.nodes():
            left_nodes = list(G.nodes())
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

    adeg_OR = sum([j for i,j in G.degree()])/G.number_of_nodes()
    print("Rel Error wrt to D %s %%" % (rhu((adeg_OR - D)/D,1 )*100))
    print("Rel Error wrt to adeg_CM %s %%" % (rhu( (adeg_OR - adeg_CM)/adeg_CM,1 )*100))
    
    return G


N = int(1e3); p_max = 0.1; add_edges_only = True

'progression of net-parameters'
k_prog = np.arange(2,18,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.001,0.5,10
    )
beta_prog = np.linspace(0.01,1,8)
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
    
    folder = f"Overlapping_Rew_Add_{add_edges_only}"

    print("N: %s, D: %s" % (N,D) ) 
    G = NN_Overlapping_Conf_Model(N, D, p = p)
    
    'plot G, degree distribution and the adiaciency matrix and save them'
    save_log_params(folder = folder, text = text)

    infos_sorted_nodes(G, num_nodes= True)
    
    if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
      plot_save_net(G = G, folder = folder, p = p, done_iterations = done_iterations)
      saved_nets.append("N%s_D%s_p%s"% (N,D,rhu(p,3)))
      #print(saved_nets)
    plot_save_sir(G, folder = folder, beta = beta, mu = mu, p = p, done_iterations = done_iterations)
