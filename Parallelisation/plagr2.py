import numpy as np
import ray

ray.init(num_cpus=2)

@ray.remote
class ListActor:
    def __init__(self, l):
        self._list = l

    def get(self, i):
        return self._list[i]

    def set(self, i, val):
        self._list[i] = val

    def to_list(self):
        return self._list

susc_list = ListActor.remote(["S" for _ in range(10)])


@ray.remote
def infect(susc_list,i):
    print(f"susc[{i}] is {ray.get(susc_list.get.remote(i))} but will change to I")
    susc_list.set.remote(i, "I")
    print(f"susc[{i}] is {ray.get(susc_list.get.remote(i))} changed")

# We need to make sure this function finishes executing before we print.
for i in range(5):
    infect.remote(susc_list,i)
#ray.get(Chicken.remote(NoZeros))

print(ray.get(susc_list.to_list.remote()))