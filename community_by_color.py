#https://stackoverflow.com/questions/65069624/networkx-cluster-nodes-in-a-circular-formation-based-on-node-color

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from definitions import save_log_params, plot_sir, check_loops_parallel_edges, remove_loops_parallel_edges


NODE_LAYOUT = nx.circular_layout
COMMUNITY_LAYOUT = nx.circular_layout


def partition_layout(g, partition, ratio=0.3):
    """
    Compute the layout for a modular graph.

    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        network to plot

    partition -- dict mapping node -> community or None
        Network partition, i.e. a mapping from node ID to a group ID.

    ratio: 0 < float < 1.
        Controls how tightly the nodes are clustered around their partition centroid.
        If 0, all nodes of a partition are at the centroid position.
        if 1, nodes are positioned independently of their partition centroid.

    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition)

    pos_nodes = _position_nodes(g, partition)
    pos_nodes = {k : ratio * v for k, v in pos_nodes.items()}

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = COMMUNITY_LAYOUT(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """
    communities = dict()
    for node, community in partition.items():
        if community in communities:
            communities[community] += [node]
        else:
            communities[community] = [node]

    pos = dict()
    for community, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = NODE_LAYOUT(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def _layout(networkx_graph):
    edge_list = [edge for edge in networkx_graph.edges]
    node_list = [node for node in networkx_graph.nodes]

    pos = nx.circular_layout(edge_list)

    # NB: some nodes might not be connected and hence will not be in the edge list.
    # Assuming a [0, 0, 1, 1] canvas, we assign random positions on the periphery
    # of the existing node positions.
    # We define the periphery as the region outside the circle that covers all
    # existing node positions.
    xy = list(pos.values())
    centroid = np.mean(xy, axis=0)
    delta = xy - centroid[np.newaxis, :]
    distance = np.sqrt(np.sum(delta**2, axis=1))
    radius = np.max(distance)

    connected_nodes = set(_flatten(edge_list))
    for node in node_list:
        if not (node in connected_nodes):
            pos[node] = _get_random_point_on_a_circle(centroid, radius)

    return pos


def _flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def _get_random_point_on_a_circle(origin, radius):
    x0, y0 = origin
    random_angle = 2 * np.pi * np.random.random_sample()
    x = x0 + radius * np.cos(random_angle)
    y = y0 + radius * np.sin(random_angle)
    return np.array([x, y])

def comm_caveman_relink(cliques = 8, clique_size = 7, p = 0,  relink_rnd = 0, relink_ring = 0):
    import numpy as np
    import numpy.random as npr
    'caveman_graph'
    G = nx.caveman_graph(l = cliques, k = clique_size)

    'relink nodes to neighbor "cave"'
    total_nodes = clique_size*cliques
    #if relink_ring != 0: 
    for clique in range(cliques):
        nodes_inclique = np.arange(D*(clique), D*(1+clique))
        if relink_ring != 0:
            tests = npr.choice(nodes_inclique, relink_ring)
            attached_nodes = npr.choice( np.arange(D*(1+clique), D*(2+clique)), 
                                        size = len(tests) )
            attached_nodes = attached_nodes % np.max((total_nodes,1))
            for test, att_node in zip(tests, attached_nodes):
                #print("NN - clique add:", (test,att_node))
                G.add_edge(test,att_node)
        if p != 0:
            tests = npr.choice(nodes_inclique, relink_rnd)
            attached_nodes = npr.choice([x for x in G.nodes() if x not in nodes_inclique], 
                                        size = len(tests))
            for test, att_node in zip(tests, attached_nodes):
                #print("relink", (test,att_node))
                if npr.uniform() < p: G.add_edge(test,att_node)

    check_loops_parallel_edges(G)
    remove_loops_parallel_edges(G)
    #check_loops_parallel_edges(G)

    print("size/cliq: %s, cliq/size: %s" % (clique_size/cliques, cliques/clique_size) )

    
    return G


'start of the main()'
from itertools import product
from definitions import rhu, plot_save_net, plot_save_sir


p_max = 0.1; N = int(1e3)

'progression of net-parameters'
k_prog = np.arange(10,32,2)
p_prog = np.linspace(0,p_max,4)
mu_prog = np.linspace(0.01,1,7)
beta_prog = np.linspace(0.01,1,7)
R0_min = 0.5; R0_max = 7

N = int(1e3)

'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []
for D,p,beta,mu in product(k_prog, p_prog, beta_prog, mu_prog):  
  if R0_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("Iterations left: %s" % ( total_iterations - done_iterations ) )
    
    folder = "Caveman_Model"
    clique_size = D; cliques = int(N/D)
    G = comm_caveman_relink(cliques=cliques, clique_size = D, 
                            p = p, relink_rnd = 1, relink_ring = 1)
    
    partition = {node : np.int(node/clique_size) for node in range(cliques * clique_size)}
    pos = partition_layout(G, partition, ratio=clique_size/cliques*0.1)

    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max)

    if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
      print("This is p", p)
      plot_save_net(G = G, pos = pos, partition = partition, folder = folder, D = D, p = p, done_iterations = done_iterations)
      saved_nets.append("N%s_D%s_p%s"% (N,D,rhu(p,3)))
      print(saved_nets)
    plot_save_sir(G, folder = folder, beta = beta, D = D, mu = mu, p = p_max, done_iterations = done_iterations)
    
    save_log_params(folder = folder, text = text)

'''
ax = plt.subplot()
plot_sir(G, ax = ax, D = D, beta = beta, mu = mu) #mf! expected
plt.show()'''