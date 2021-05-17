import numpy as np
from itertools import product
from definitions import save_log_params, plot_save_nes, \
config_pois_model, NN_pois_net, parameters_net_and_sir

'save scaled version for better visualization'
def scaled_conf_pois(G,D,cut_off=30):    
  scaled_N = int(G.number_of_nodes()/cut_off) # int(D/cut_off)
  return config_pois_model(scaled_N, D)

'''Configurational Model with poissonian degree:
1) Question: 
  1. using this meth, loops and parallel edges are present leading 
  to a adj_matrix not normalized to 1. Since neighbors are involved in sir 
  and contagious is made by (fut = S, curr = S) - nodes, I may leave them, 
  but there're not so in line of "social contacts". 
  So, I prefer to remove them, but I loose a lot of precision if $D / N !<< 1$. 
  <br>Ex., $D = 50 = N, <k> ~ 28$. For $N= 1000 \text{ and } D = 3 \textrm{ or } 8, 
  <k> \textrm{is acceptable.}$
'''
N = int(1e3); p_max = 0;  folder = "NNR_Conf_Model"

'progression of net-parameters'
'''
k_prog = np.arange(2,34,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.1,1,15)
beta_prog = np.linspace(0.1,1,15)
p_prog = [0]
R0_min = 0.5; R0_max = 6
'''

k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max =  parameters_net_and_sir(folder = folder, p_max = p_max) 

'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

'save parameters'
text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s; \nTotal Iterations: %s;\n---\n" \
        % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
        mu_prog, len(mu_prog),  R0_min, R0_max, total_iterations)
save_log_params(folder = folder, text = text)

saved_nets = []
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("\nIterations left: %s" % ( total_iterations - done_iterations ) )
    
    done_iterations = 0; saved_nets = []
    for D,p in product(k_prog, p_prog):  
      for beta, mu in zip(beta_prog, mu_prog):
        'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
        if R0_min < beta*D/mu < R0_max and beta <= 1:
          done_iterations+=1
          print("Iterations left: %s" % ( total_iterations - done_iterations ) )

          G = NN_pois_net(N, D = D)
          plot_save_nes(G = G, 
          p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
          plot_save_nes(G = G,
          p = p, folder = folder, adj_or_sir="SIR", beta = beta, mu = mu, done_iterations=done_iterations)
          print("---")