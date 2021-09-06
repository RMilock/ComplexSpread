import numpy as np
from multiprocessing import Pool

arr = ['a','b','c','d','e','f','g']

def edit_array(array_2D, i, attribute):
    print(array_2D)
    return arr[i] + attribute

if __name__=='__main__':
    list_start_vals = range(len(arr))
    array_2D = []
    with Pool(processes=4) as f:
        array_2D = f.starmap(edit_array, \
            [(arr, i, " hello") for i in list_start_vals])
    print(f"array_2D: {array_2D}")