import json
import numpy as np
from collections import defaultdict

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {float(k):v for k,v in x.items()}
    return x

class NestedDict(dict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]

def getkeys(obj, stack):
    for k, v in obj.items():
      k2 = [k] + stack # stack, i.e. already present -- return 0.0 keys
      #print("\nk,v, stack,k2", k,v, stack, k2)
      #print(type(v), isinstance(v, dict))
      #if not v: print("if v is False")
      #if v: print("if v is True")
      if v and isinstance(v, dict):
        #print("getkeys", list(getkeys(v,k2)) )
        for c in getkeys(v, k2):
          #print("yield c", c)
          yield c[::-1]
      else: # leaf
        #print("yield k2")
        yield k2


path = "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/NN_Conf_Model/Std/"
dic_path = "".join( (path, "saved_std_dicts.txt") )

d = NestedDict() 
print(type(d), )

with open(dic_path,"r") as f:
    d = json.loads(f.read(), object_hook=jsonKeys2int) # type(d) = dic

old_keys = list(getkeys(d, []))

print("bef filling", d, type(d), old_keys)

#print(json.dumps(d, sort_keys=False, indent=4) ) # type(d) = string

d = NestedDict(d)

p,m,b,D = 1000, 0.05, 0.05, 4
p,m,b,D = 0.0, 0.02, 0.05, 4
new_keys = [p,m,b,D]

value = 34
if p in d.keys():
    if m in d[p].keys():
        if b in d[p][m].keys():
            d[p][m][b][D] = value
        else: d[p][m] = { **d[p][m], **{b: {D:value}} }
    else: d[p] = {**d[p], **{m:{D:value}}}
else:
    d[p][m][b][D] = 34

print(d)


for i in np.arange(len(old_keys)):
    if new_keys == old_keys[i]:
        print("already exists!", new_keys, old_keys)
        new_dic = d[p][m][b][D].append(35)
        #d = new_dic
        print(json.dumps(new_dic, sort_keys=False, indent=4)) # type(d) = string
        
    else: 
        print("don't exist!", new_keys, old_keys); 
        d[p][m][b][D] = 44
'''
d = json.dumps(d, sort_keys=False, indent=4) # type(d) = string

d[1000][0.05][0.05][4] = 4
d[0.0][0.05] = 34

print("d and type d after reimpitura:\n", d, type(d)) 

new_filename = "".join( (path, "pp_save_std_dicts.txt") )

d = json.dumps(d, sort_keys=False, indent=4) # type(d) = string

with open(new_filename,"w") as f:
    f.write(d)'''