from math import inf
import tvm
import os
import pickle
import numpy as np
import multiprocessing as mp

def get_divisors(number):
    divisors = []
    for i in range(2, int(number/2), 1):
        #print(i)
        if number % i == 0:
            divisors.append(i)
    
    return divisors

def load_last_state(state_file_path, worker):
    state_file_path += worker
    if not os.path.exists(state_file_path):
        print("[STATE FILE] NOT FOUND")
        return []
    
    last_state = pickle.load( open(state_file_path, "rb") )
    return last_state

def write_state(state_path, state_name, state, worker=""):
    if not os.path.exists(state_path):
        folders = state_path.split("/")
        prev_folder = ""
        for folder in folders:
            prev_folder = prev_folder + "/" + folder
            os.mkdir(prev_folder)
    
    pickle.dump(state, open(state_path+"/"+state_name+str(worker), "wb"))
    return

def create_empty_state():
    state = {
        "n" : 0,
        "c" : 0,
        "h" : 0,
        "w" : 0,
        "kernel" : 0,
        "strides" : 0,
        "pad" : 0,
        "dil" : 0,
        "grps" : 0,
        "channels" : 0,
        "units" : 0,
        "inp" : 0,
        "layer" : None,
        "setup" : False,
    }
    return state


def get_metrics(target_name, device_key, backend="nvml", dev_idx=0):
    # written like this to accommodate multiple metric readings in the future, if necessary
    if backend == "nvml":
        metrics = []
        if "cuda" in target_name:
            if device_key == "980ti":
                metrics.append(
                    "nvml:::NVIDIA_GeForce_GTX_980_Ti:device_"+str(dev_idx)+":power")
            if device_key == "1660ti":
                metrics.append(
                    "nvml:::NVIDIA_GeForce_GTX_1660_Ti:device_"+str(dev_idx)+":power")

            if device_key == "2060":
                metrics.append("nvml:::NVIDIA_GeForce_RTX_2060:device_"+str(dev_idx)+":power")
            if device_key == "2080ti":
                metrics.append(
                    "nvml:::NVIDIA_GeForce_RTX_2080_Ti:device_"+str(dev_idx)+":power")
            if device_key == "K80":
                metrics.append(
                    "nvml:::Tesla_K80:device_"+str(dev_idx)+":power")   
            if device_key == "A100":
                metrics.append("nvml:::NVIDIA_A100-SXM4-40GB:device_"+str(dev_idx)+":power")

            return metrics
        else:
            metrics = [
                "rapl:::PACKAGE_ENERGY:PACKAGE0",
                "rapl:::DRAM_ENERGY:PACKAGE0"
            ]

            return metrics
    if backend == "visa":
        return [":MEAS:SCAL:VOLT:DC? CH1,(@1)", ":MEAS:SCAL:CURR:DC? CH1,(@1)"]
    return []

'''
def get_collector(dev, metrics, backend="visa"):
    if backend == "visa":
        profile_collector = 
        )

    if backend == "nvml":
        profile_collector = tvm.runtime.profiling.PAPIMetricCollector({
                                                                      dev: metrics})

    return profile_collector

'''
def evaluate_power(data):
    metrics = data.keys()
    power = 1
    found = 0
    for metric in metrics:
        if found == 2:
            return power

        if ":CURR:DC?" in metric:
                power *= data[metric].value
                found += 1 
        if ":VOLT:DC?" in metric:
            power *= data[metric].value
            found += 1
    if found == 2:
        return power
    return int("inf")

### RANDOM SEARCH

def pretty_print_pool_search_space(search_space : dict):
    print(" [TOTAL SEARCH SPACE]")
    print("  [N]")
    print("    MIN:  ", search_space["range_n"][0])
    print("    MAX:  ", search_space["range_n"][1]-1)
    print("    STEP: ", search_space["range_n"][2])
    print()

    print("  [H]")
    print("    MIN:  ", search_space["range_h"][0])
    print("    MAX:  ", search_space["range_h"][1]-1)
    print("    STEP: ", search_space["range_h"][2])
    print()

    print("  [W]")
    print("    MIN:  ", search_space["range_w"][0])
    print("    MAX:  ", search_space["range_w"][1]-1)
    print("    STEP: ", search_space["range_w"][2])
    print()

    print("  [C]")
    print("    MIN:  ", search_space["range_c"][0])
    print("    MAX:  ", search_space["range_c"][1]-1)
    print("    STEP: ", search_space["range_c"][2])
    print()

    print("  [POOL]")
    print("    MIN:  ", search_space["range_pool"][0])
    print("    MAX:  ", search_space["range_pool"][1]-1)
    print("    STEP: ", search_space["range_pool"][2])
    print()

    print("  [STRIDE]")
    print("    MIN:  ", search_space["range_strides"][0])
    print("    MAX:  ", search_space["range_strides"][1]-1)
    print("    STEP: ", search_space["range_strides"][2])
    print()

    print("  [PADDING]")
    print("    MIN:  ", search_space["range_pad"][0])
    print("    MAX:  ", search_space["range_pad"][1]-1)
    print("    STEP: ", search_space["range_pad"][2])
    print()

    print("  [DILATION]")
    print("    MIN:  ", search_space["range_dil"][0])
    print("    MAX:  ", search_space["range_dil"][1]-1)
    print("    STEP: ", search_space["range_dil"][2])
    print()
    return

def pretty_print_dense_search_space(search_space : dict):
    print(" [TOTAL SEARCH SPACE]")
    print("  [N]")
    print("    MIN:  ", search_space["range_n"][0])
    print("    MAX:  ", search_space["range_n"][1]-1)
    print("    STEP: ", search_space["range_n"][2])
    print()

    print("  [INP]")
    print("    MIN:  ", search_space["range_inp"][0])
    print("    MAX:  ", search_space["range_inp"][1]-1)
    print("    STEP: ", search_space["range_inp"][2])
    print()

    print("  [UNITS]")
    print("    MIN:  ", search_space["range_units"][0])
    print("    MAX:  ", search_space["range_units"][1]-1)
    print("    STEP: ", search_space["range_units"][2])
    print()

    return

## CONV2D

def pretty_print_conv2d_search_space(search_space : dict):
    print(" [TOTAL SEARCH SPACE]")
    print("  [N]")
    print("    MIN:  ", search_space["range_n"][0])
    print("    MAX:  ", search_space["range_n"][1]-1)
    print("    STEP: ", search_space["range_n"][2])
    print()

    print("  [H]")
    print("    MIN:  ", search_space["range_h"][0])
    print("    MAX:  ", search_space["range_h"][1]-1)
    print("    STEP: ", search_space["range_h"][2])
    print()

    print("  [W]")
    print("    MIN:  ", search_space["range_w"][0])
    print("    MAX:  ", search_space["range_w"][1]-1)
    print("    STEP: ", search_space["range_w"][2])
    print()

    print("  [C]")
    print("    MIN:  ", search_space["range_c"][0])
    print("    MAX:  ", search_space["range_c"][1]-1)
    print("    STEP: ", search_space["range_c"][2])
    print()

    print("  [KERNEL]")
    print("    MIN:  ", search_space["range_kernel"][0])
    print("    MAX:  ", search_space["range_kernel"][1]-1)
    print("    STEP: ", search_space["range_kernel"][2])
    print()

    print("  [STRIDE]")
    print("    MIN:  ", search_space["range_strides"][0])
    print("    MAX:  ", search_space["range_strides"][1]-1)
    print("    STEP: ", search_space["range_strides"][2])
    print()

    print("  [PADDING]")
    print("    MIN:  ", search_space["range_pad"][0])
    print("    MAX:  ", search_space["range_pad"][1]-1)
    print("    STEP: ", search_space["range_pad"][2])
    print()

    print("  [DILATION]")
    print("    MIN:  ", search_space["range_dil"][0])
    print("    MAX:  ", search_space["range_dil"][1]-1)
    print("    STEP: ", search_space["range_dil"][2])
    print()

    print("  [GROUPS]")
    print("    MIN:  ", search_space["range_grps"][0])
    print("    MAX:  ", search_space["range_grps"][1]-1)
    print("    STEP: ", search_space["range_grps"][2])
    print()

    print("  [FILTER COUNT]")
    print("    MIN:  ", search_space["range_channels"][0])
    print("    MAX:  ", search_space["range_channels"][1]-1)
    print("    STEP: ", search_space["range_channels"][2])
    print()
    return

def extend_search_space(search_space : dict):
    extended = {}
    
    for key, value in search_space.items():
        start = value[0]
        end = value[1]
        steps = value[2]
        if len(value) > 3:
            first = start
            start = value[3]
            
        extended[key] = list(range(start, end+1, steps))
        
        if len(value) > 3:
            while extended[key][0] < first:
                del extended[key][0]
            extended[key].insert(0,first)
        
    return extended

def rand_tensor_shape(extended_dims : list, rng=None, max_size=1*(1024**3)):
    if rng == None:
        rng = np.random.default_rng()
    
    total = inf
    shape = []
    
    while total > max_size:
        shape = []
        for dim in extended_dims:
            shape.append(dim[rng.integers(0, len(dim))])
        total = np.prod(shape)*4
    
    return shape    

def new_rand_conv2d(extended_search_space : dict, rng = None, grouped : bool = True, dilated : bool = True, max_in_size=41943040, max_w_size=20971520, event = None):
    if rng == None:
        rng = np.random.default_rng()
    if event != None:
        print("MP")
        
    extended = extended_search_space
        
    config = {}
    #create input tensor
    n, c, h, w = rand_tensor_shape([extended["range_n"],
                                  extended["range_c"],
                                  extended["range_h"],
                                  extended["range_w"]
                                 ],
                                rng=rng,
                                max_size=max_in_size)
    config["n"] = n
    config["c"] = c
    config["h"] = h
    config["w"] = w
    
    ##todo: padding, strides, dilation
    config["strides"] = extended["range_strides"][rng.integers(0, len(extended["range_strides"]))]
    config["pad"] = extended["range_pad"][rng.integers(0, len(extended["range_pad"]))]
    if dilated:
        config["dilation"] = extended["range_dil"][rng.integers(0, len(extended["range_dil"]))]
    else:
        config["dilation"] = 1
        
    if grouped:
        '''grp_steps = 1
        if len(extended["range_grps"]) >= 3:
            grp_steps = extended["range_grps"][-1] - extended["range_grps"][-2]
        #print(grp_steps)

        grp_space = list(range(0, 1+min(config["c"], extended["range_grps"][-1]), grp_steps))'''
        grp_space = [1]
        for i in extended["range_grps"]:
            if c%i == 0:
                grp_space.append(i)
        grp_space[0] = 1
    else:
        grp_space = [1]
    #print(grp_space)
            
    kernel_space = [1]
    for i in extended["range_kernel"]:
        if i <= (config["w"]+config["pad"])/config["dilation"] and i <= (config["h"]+config["pad"])/config["dilation"]:
            kernel_space.append(i)
        else:
            break
    #create weight tensor
    total = inf
    while total > max_w_size:
        kernel = kernel_space[rng.integers(0, len(kernel_space))]
        if grouped:
            grps = grp_space[rng.integers(0, len(grp_space))]
        else:
            grps = 1
            
        grps_size = int(config["c"] / grps)
        
        filter_options = [grps_size]
        for i in extended["range_channels"]:
            if i%grps==0:
                filter_options.append(i)
        
        filters = filter_options[rng.integers(0, len(filter_options))]
        
        total = kernel**2 * c/grps * filters * 4
    
    config["kernel"] = kernel
    config["grps"] = grps
    config["channels"] = filters
    
    if config["pad"] > config["kernel"]*config["dilation"]:
        config["pad"] = int(config["kernel"]/2)
    
    if event != None:
        event.set()        
    return config

def get_rand_pool(search_space : dict, dilated=True, max_in_size=2*(1024**3)):
    config = {}

    range_n = search_space["range_n"]
    range_h = search_space["range_h"]
    range_w = search_space["range_w"]
    range_c = search_space["range_c"]

    range_pad = search_space["range_pad"]
    range_dil = search_space["range_dil"]
    range_pool = search_space["range_pool"]
    range_strides = search_space["range_strides"]

    total_size = inf
    while total_size > max_in_size:
        lower = np.floor(range_n[0]/range_n[2])
        high = np.ceil(range_n[1]/range_n[2])
        config["n"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_n[2], range_n[0]]))

        # H
        lower = np.floor(range_h[0]/range_h[2])
        high = np.ceil(range_h[1]/range_h[2])
        config["h"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_h[2], range_h[0]]))

        # W
        lower = np.floor(range_w[0]/range_w[2])
        high = np.ceil(range_w[1]/range_w[2])
        config["w"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_w[2], range_w[0]]))

        # C
        lower = np.floor(range_c[0]/range_c[2])
        high = np.ceil(range_c[1]/range_c[2])
        config["c"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_c[2], range_c[0]]))

        total_size = config["n"] * config["c"] * config["h"] * config["w"] * 4 # assuming float32
    
    # PAD
    lower = np.floor(range_pad[0]/range_pad[2])
    high = np.ceil(range_pad[1]/range_pad[2])
    config["pad"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_pad[2], range_pad[0]]))

    # DILATION
    config["dilation"] = 1
    if dilated:
        lower = np.floor(range_dil[0]/range_dil[2])
        high = np.ceil(range_dil[1]/range_dil[2])
        config["dilation"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_dil[2], range_dil[0]]))
    
    # KERNEL
    lower = np.ceil(range_pool[0]/range_pool[2])
    high = np.ceil(
            np.min([
                range_pool[1],
                int((config["w"]+config["pad"])/config["dilation"]),
                int((config["h"]+config["pad"])/config["dilation"])])
            /range_pool[2])
    try:
        config["pool"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_pool[2], range_pool[0]]))
    except:
        print("TODO: fix generation of pool size")
        config["pool"] = int(lower)

    # STRIDES
    lower = np.floor(range_strides[0]/range_strides[2])
    high = np.ceil(range_strides[1]/range_strides[2])
    config["strides"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_strides[2], range_strides[0]]))

    return config

def get_rand_dense(search_space : dict, max_in_size=20971520, max_w_size=20971520):
    config = {}

    range_n = search_space["range_n"]
    range_inp = search_space["range_inp"]
    range_units = search_space["range_units"]

    total_inp_size = inf
    total_w_size = inf
    while total_inp_size >max_in_size and total_w_size >max_w_size:
        config["n"] = int(np.random.randint(range_n[0], range_n[1], dtype="int"))
        config["inp"] = int(np.random.randint(range_inp[0], range_inp[1], dtype="int"))
        config["units"] = int(np.random.randint(range_units[0], range_units[1], dtype="int"))

        total_inp_size = config["n"] * config["inp"] * 4
        total_w_size = config["n"] * config["inp"] * 4

    return config

def get_rand_conv2d(search_space : dict, grouped : bool, dilated : bool, max_in_size=41943040, max_w_size=20971520):

    config = {}
    # what are the dependencies between hyperparameters?
    # N 

    range_n = search_space["range_n"]
    range_h = search_space["range_h"]
    range_w = search_space["range_w"]
    range_c = search_space["range_c"]

    range_pad = search_space["range_pad"]
    range_dil = search_space["range_dil"]
    range_kernel = search_space["range_kernel"]
    range_strides = search_space["range_strides"]
    range_grps = search_space["range_grps"]
    range_channels = search_space["range_channels"]

    total_size = inf

    while total_size > max_in_size:
        lower = np.floor(range_n[0]/range_n[2])
        high = np.ceil(range_n[1]/range_n[2])
        config["n"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_n[2], range_n[0]]))

        # H
        lower = np.floor(range_h[0]/range_h[2])
        high = np.ceil(range_h[1]/range_h[2])
        config["h"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_h[2], range_h[0]]))

        # W
        lower = np.floor(range_w[0]/range_w[2])
        high = np.ceil(range_w[1]/range_w[2])
        config["w"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_w[2], range_w[0]]))

        # C
        lower = np.floor(range_c[0]/range_c[2])
        high = np.ceil(range_c[1]/range_c[2])
        config["c"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_c[2], range_c[0]]))

        total_size = config["n"] * config["c"] * config["h"] * config["w"] * 4 # assuming float32

    # kernel*strides should be smaller than h (+pad) or w (+pad)
    # dilation is also involved here

    # PAD
    lower = np.floor(range_pad[0]/range_pad[2])
    high = np.ceil(range_pad[1]/range_pad[2])
    config["pad"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_pad[2], range_pad[0]]))

    # DILATION
    config["dilation"] = 1
    if dilated:
        lower = np.floor(range_dil[0]/range_dil[2])
        high = np.ceil(range_dil[1]/range_dil[2])
        config["dilation"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_dil[2], range_dil[0]]))

    total_size = inf
    while total_size > max_w_size:
        # KERNEL
        lower = np.ceil(range_kernel[0])
        high = np.ceil(
                np.min([
                    range_kernel[1],
                    int((config["w"]+config["pad"])/config["dilation"]),
                    int((config["h"]+config["pad"])/config["dilation"])])
                )
        try:
            config["kernel"] = int(np.max([np.random.randint(lower, high, dtype="int"), range_kernel[0]]))
        except:
            print("TODO: fix generation of kernel size")
            config["kernel"] = int(lower)

        # STRIDES
        lower = np.floor(range_strides[0]/range_strides[2])
        high = np.ceil(range_strides[1]/range_strides[2])
        config["strides"] = int(np.max([np.random.randint(lower, high, dtype="int")*range_strides[2], range_strides[0]]))

        # GRPS
        config["grps"] = 1
        if grouped and not config["c"] == 1:
            div_of_c = get_divisors(config["c"])
            if len(div_of_c) == 0:
                config["grps"] = 1
            else:
                config["grps"] = int(config["c"] / div_of_c[np.random.randint(0, len(div_of_c)-1)])    
        if config["grps"] == 0:
            config["grps"] = 1
            
        # OUT CHANNELS
        lower = np.floor(range_channels[0]/range_channels[2])
        high = np.ceil(range_channels[1]/range_channels[2])
        config["channels"] = config["grps"] * int(np.max([np.random.randint(lower, high, dtype="int")*range_channels[2], range_channels[0]]))

        if grouped:
            config["channels"] = int(config["c"] / config["grps"]) * np.random.randint(1, 12)
        total_size = config["kernel"] * config["kernel"] *(config["c"] / config["grps"]) * config["channels"]*4

    return config