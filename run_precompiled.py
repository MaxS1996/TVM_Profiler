from numpy.core.numeric import full
import tvm
from tvm.contrib import utils, graph_executor as runtime
from tvm.relay.op.nn.nn import conv2d, dilate
#####
import numpy as np
import os
import time
import platform
import sys
#from math import prod
import json
import pickle
### pip install func-timeout
from func_timeout import func_timeout, FunctionTimedOut
#####
from profile_config import *

print("exploring precompiled modules from ", module_path+"/")
print()

module_options = set(os.listdir(module_path))
print(len(module_options), "found")
print()

state = {}
full_state_path = full_state_path+"_"+"precomp"
state_file += "_"+"precomp"
if os.path.exists(full_state_path):
    state = pickle.load(open(full_state_path, "rb"))
    print("found state")
    print(len(state["done"]), "already profiled")
else:
    state = {}
    state["done"] = set()
    pickle.dump(state, open(full_state_path, "wb"))

module_options = module_options.difference(state["done"])

for option in module_options:
    state["done"].add(option)
    pickle.dump(state, open(full_state_path, "wb"))
    
    full_path = module_path+"/"+option+"/"
    print(full_path)
    files = os.listdir(full_path)

    inp_shape = None
    inp_data = None
    params = None
    graph_json = None
    module = None

    layer_name = None
    description_vecs = {}
    metas = {}
    memory_data = {}
    config = None
    conf_str = None

    if not "data.params" in files:
        print(full_path, "lacks data.params file")
        continue

    if not "compressed_input.data" in files:
        print(full_path, "lacks input tensor data")
        continue

    if not "config.json" in files:
        print(full_path, "lacks config.json file")
        continue

    if not "graph.json" in files:
        print(full_path, "lacks graph.json file")
        continue

    if not "graph.json" in files:
        print(full_path, "lacks lib.tar file")
        continue

    for file in files:
        if file == "data.params":
            params = pickle.load(open(full_path+file, "rb"))
            print("found params")
            continue
        
        if file == "compressed_input.data":
            raw_input_data, inp_shape = pickle.load(open(full_path+file, "rb"))
            print("found compressed input data")
            print("expanding inputs")
            required = int(np.prod(inp_shape))
            rand_repeat = int(np.prod(inp_shape)/np.prod(raw_input_data.shape))
            inp_data = np.repeat(raw_input_data, rand_repeat)[:required].reshape(inp_shape).astype("float32")
            del raw_input_data
            continue
            
        if file == "config.json":
            config = json.loads(open(full_path+file, "rb"))
            print("found config")
            continue

        if file == "graph.json":
            with open(full_path+file, "r") as f:
                graph_json = f.read()
            print("found graph JSON")
            continue

        if file == "lib.tar":
            module = tvm.runtime.load_module(full_path+file)
            print("found module")
            continue

    print("LOADING MODULE COMPLETE")

    from tvm.contrib.debugger import debug_executor as graph_runtime

    debug_g_mod = graph_runtime.GraphModuleDebug(
        module["debug_create"]("default", dev),
        [dev],
        graph_json,
        ".",
    )

    try:
        debug_g_mod.set_input("data", tvm.nd.array(inp_data.astype("float32")))
        #func_timeout(execution_timeout, debug_g_mod.run, None, None)
        raw_times = debug_g_mod.run_individual(number=3, repeat=3, min_repeat_ms=300)
    except:
        print("[WARNING] EXECUTION FAILED")
        continue
    print("[INFO] EXECUTION COMPLETED")            
    times = {}
    for idx, node in enumerate(debug_g_mod.debug_datum._nodes_list):
        times[node["op"]] = raw_times[idx]
        if layer_name in node["op"]:
            layer_time = raw_times[idx]
            actual_func_name = node["op"]
    print(layer_time)
    config["time"] = layer_time
    print("[INFO] EXTRACTED TIME INFORMATION")
    
    if data_collector != None:
        # measure power consumption
        runs = max(1000, int(time_min_res / layer_time))
        print(runs, "repetitions required")

        try:
            debug_g_mod.set_input("data", tvm.nd.array(inp_data.astype("float32")))
            print("input prepared")
            test_data = debug_g_mod.profile(collectors=[data_collector], data=tvm.nd.array(inp_data), runs=runs)
            print("profiling done")
        except:
            print("[WARNING] PROFILING FAILED")
            continue
        print("[INFO] PROFILING COMPLETED")
        # store everything
        powers = {}
        for idx, data in enumerate(test_data.calls):
            print(idx)
            print(data)
            powers[data["Name"]] = profiling.evaluate_power(dict(data))
            print()
    else:
        powers = {}
        for layer in description_vecs.keys():
            powers[layer] = -1
            print()

    config["power"] = powers[actual_func_name]

    ## store result
    with open(data_path+"/"+target+"_"+option, "w") as f:
        f.write(json.dumps(config))
    #serializer.dump_description_vectors(description_vecs, metas, conf_str, device+"_"+target, "serialized"+"_"+layer_name, path="./dataset")

    print("RUN COMPLETED")


    
