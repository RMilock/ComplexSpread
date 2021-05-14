
string = "AdjMat_N:%s, k_{max}: %s" %(10000,3)
folder = "AdjMat"
print(string)

string = string.strip("".join((folder,"_")))
string = "".join(("r\"$", string,"$\""))
print(string, eval(string))

folder = "B-A_Model"
N = 1000; D = 3; p = 0.0; adj_or_sir = "AdjMat"; max_degree = 3; m, N0 = 1,1

def func_file_name(folder, adj_or_sir, N, D, p, max_degree, m = 0, N0 = 0, beta = 0.111, mu = 1.111):
  from definitions import rhu
  max_degree = 0
  if adj_or_sir == "AdjMat":
    print( adj_or_sir, N, D, rhu(p,3), max_degree, m, N0 )
    if folder == "B-A_Model": 
      name = folder + "_%s_N%s_D%s_p%s_k_max%s_m%s_N0_%s" % (
      adj_or_sir, N, D, rhu(p,3), max_degree, m, N0) + \
        ".png"
      print("name", name)
      return name
    else: return folder + "_%s_N%s_D%s_p%s.png" % (adj_or_sir, N,rhu(D,1),rhu(p,3)) 

  if adj_or_sir == "SIR":
    return folder + "_%s_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% (
            adj_or_sir, '{:.3f}'.format(rhu(beta/mu*D,3)),
            N,D, rhu(p,3), rhu(beta,4), rhu(mu,3) ) + ".png"

file_name = func_file_name(folder, adj_or_sir, N, D, p, max_degree, m, N0)

print(file_name)