# IMPORTS from tvm, different required packages and the profiling infrastructure

from copy import deepcopy
import tvm
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.debugger import debug_executor as graph_runtime
from tvm.relay.op.nn.nn import dense, dilate, conv2d, avg_pool2d, max_pool2d
#####
import numpy as np
import pynvml as nv
from func_timeout import func_timeout
import time
import json
import sys
#####
from components import description_vector as dv
from components import serializer
from components import profiling

#####################################################
dataset_path = "/home/s0144002/DIR/ssd/s0144002-TVMMapper/TVM_Profiling_Dataset/dataset"
#dataset_path = "/home/s0144002/DIR/ssd/s0144002-TVMMapper/TVM_Profiler/testing"
#####################################################

import sys, getopt
import argparse
workload = "conv2d"
workload_options = [
    "conv2d",
    "dilated_conv2d",
    "depthwise_conv2d",
    "pool2d",
    "dense",
    "test_set",
    ]

supported_targets = [
    "alpha",
    "haswell",
    "gpu2"
]

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument(
        "-t",
        "--target",
        default="980ti",
        help="The target device, you want to compile for and profile on.")
    parser.add_argument(
        "-w",
        "--workload",
        default="conv2d",
        help="The layer type you want to profile (conv2d, dilated_conv2d, depthwise_conv2d, max_pool2d, avg_pool2d, dense.")
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="An additional input file."
        )
    options = parser.parse_args(args)
    return options

options = getOptions()
print(options)

partition = options.target
if not partition in supported_targets:
    print("error, unknown partition")
print("PARTITION:", partition)

workload = options.workload
if not workload in workload_options:
    print("error, unknown workload")
print("workload:", workload)

input_path = options.input
if not input_path is None:
    print("Additional Input File:",input_path)

#####################################################
iterations = 10
#####################################################
if partition == "alpha":
    from config_alpha import *

elif partition == "haswell":
    from config_haswell import *

elif partition == "gpu2":
    from config_gpu2 import *

elif partition == "980ti":
    from config_980ti import *


samples_base_path = "./configs"
is_test_set = False
if workload == "conv2d":
    workload_paths = ["conv_layer_config_clean.json"]
    layer_name = "conv2d"
elif workload == "depthwise_conv2d":
    workload_paths = [
        "depthwise_conv_layer_config_clean.json",
        "depthwise_8_conv_layer_config_clean.json",
        "depthwise_16_conv_layer_config_clean.json",
        "depthwise_32_conv_layer_config_clean.json",
        "depthwise_128_conv_layer_config_clean.json",
        "depthwise_256_conv_layer_config_clean.json",
        "depthwise_512_conv_layer_config_clean.json",
    ]
    layer_name = "conv2d"
elif workload == "dilated_conv2d":
    workload_paths = [
        "dilated_2_conv_layer_config_clean.json",
        "dilated_4_conv_layer_config_clean.json",
        "dilated_8_conv_layer_config_clean.json",
        ]
    layer_name = "conv2d"
elif workload == "dense":
    workload_paths = [
        "dense_layer_config_clean.json",
        "rand_dense.json",
    ]
    layer_name = "dense"
elif workload == "pool2d":
    workload_paths = [
        "avg_pool_layer_config_clean.json",
        "max_pool_layer_config_clean.json",
        "rand_pool.json",
        ]
elif workload == "test_set":
    print("test set is going to be profiled")
    is_test_set = True
    samples_base_path = "./test_set"
    workload_paths = [
        "alexnet_conv_configs.json",
        "alexnet_dense_configs.json",
        "alexnet_max-pool_configs.json",
        #"alexnet_unknown_configs.json",

        "darknet-19_conv_configs.json",
        "darknet-19_dense_configs.json",
        "darknet-19_max-pool_configs.json",
        #"darknet-19_unknown_configs.json",

        "mnist_net_conv_configs.json",
        "mnist_net_dense_configs.json",
        "mnist_net_max-pool_configs.json",
        #"mnist_net_unknown_configs.json",
    ]
if not input_path is None:
    workload_paths.append(input_path)


print(metrics)


# helpful to suppress output of debug runtime run function
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

if partition != "haswell":
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)

####################################################

import json
configs = {}
for workload_path in workload_paths:
    with open(samples_base_path+"/"+workload_path) as file:
        configs.update(json.load(file))
        print(len(configs))
print(len(configs))

print(configs)

'''
if workload == "dense":
    #expand search space
    print("artifically extending search space")
    new_configs = {}
    for name, config in configs.items():
        new_config = deepcopy(config)
        for i in dense_extension:
            new_config["units"] = new_config["output shape"][1] = config["units"] // i
            new_configs[name+"_"+str(new_config["units"])] = deepcopy(new_config)
            new_config["units"] = new_config["output shape"][1] = config["units"] * i
            new_configs[name+"_"+str(new_config["units"])] = deepcopy(new_config)
    configs = new_configs
    print("new sample count:", len(configs))
'''

measurements = {}

sample_idx = 0
for name, config in configs.items():
    print(config["output shape"])
    
    for batch_size in batch_sizes:
        sample_idx = sample_idx + 1
        print()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(sample_idx, name, batch_size)

        workload = config["workload"]
        if "pool" in workload:
            layer_name = workload

        if is_test_set:
            layer_name = config["workload"]
            file = dataset_path+"/"+target_class+"_"+device+"/test_set/"+workload + "/" + name.replace(":", "-").replace("/","_")+"_"+str(batch_size)+".json"
        else:
            file = dataset_path+"/"+target_class+"_"+device+"/"+workload + "/" + name.replace(":", "-").replace("/","_")+"_"+str(batch_size)+".json"
        folders = file.split("/")[0:-1]
        tmp = folders[0]
        
        for folder in folders[1::]:
            tmp = tmp + "/" + folder
            if not os.path.exists(tmp):
                os.mkdir(tmp)

        if os.path.exists(file):
            print("already profiled")
            break

        required = int(np.prod(config["input shape"][1::])*batch_size)

        if "pool" in workload or workload in ["conv2d", "depthwise_conv2d", "dilated_conv2d"]:
            inp_shape = (
                int(batch_size),
                int(config["input shape"][3]),
                int(config["input shape"][2]),
                int(config["input shape"][1]),
            )
        elif workload == "dense":
            inp_shape = (
                int(batch_size),
                int(config["input shape"][1]),
            )
        
        rand_data = np.random.rand(int(np.ceil(required/repeat)))
        inp_data = np.repeat(rand_data, repeat)[:required].reshape(inp_shape).astype("float32")
        x = tvm.relay.var("data", tvm.relay.TensorType(inp_shape), dtype="float32")
        
        if workload in ["conv2d", "depthwise_conv2d", "dilated_conv2d"]:
            weight_shape = (
                int(config["output shape"][3]),
                int(config["input shape"][3]//config["groups"]),
                int(config["kernel"][0]),
                int(config["kernel"][1])
            )
        elif workload == "dense":
            weight_shape = (
                int(config["units"]),
                int(config["input shape"][1]),
            )
        elif "pool" in workload:
            weight_shape = None
        
        if not "pool" in workload:
            required = int(np.prod(weight_shape))
            rand_data = np.random.rand(int(np.ceil(required/repeat)))
            weight_data = np.repeat(rand_data, repeat)[:required].reshape(weight_shape).astype("float32")
            y = tvm.relay.Constant(tvm.nd.array(weight_data))

        if workload in ["conv2d", "depthwise_conv2d", "dilated_conv2d"]:
            expr = conv2d(
                    data = x,
                    weight= y,
                    strides=config["strides"],
                    padding=0,
                    dilation=config["dilation"],
                    groups=int(config["groups"]),
                    channels=int(config["filters"]),
                    kernel_size=config["kernel"],
                    data_layout=data_layout,
                    kernel_layout=kernel_layout,
                )

        elif workload == "dense":
            expr = dense(
                    data = x,
                    weight= y,
                    units = config["units"],
                    out_dtype = config["compute dtype"]
                )

        if "pool2d" in workload:
            if not "strides" in config.keys():
                config["strides"] = config["stride"]

        if workload == "avg_pool2d":
            expr = avg_pool2d(
                data=x,
                pool_size=config["pool_size"],
                strides=config["strides"],
                padding=0,
                dilation=1,
                layout=data_layout,
                out_layout=data_layout
            )

        elif workload == "max_pool2d":
            expr = max_pool2d(
                data=x,
                pool_size=config["pool_size"],
                strides=config["strides"],
                padding=0,
                dilation=1,
                layout=data_layout,
                out_layout=data_layout
            )
        
        try:
            mod = tvm.ir.IRModule.from_expr(expr)
            params = {}
            with tvm.transform.PassContext(opt_level=3):
                compiled_graph_lib = tvm.relay.build_module.build(mod, target_class, params=params)
        except:
            print("compile failed")
            break
 
        debug_g_mod = graph_runtime.GraphModuleDebug(
            compiled_graph_lib["debug_create"]("default", dev),
            [dev],
            compiled_graph_lib.get_graph_json(),
            "."
        )
        
        if workload != "dense":
            out_shape = debug_g_mod.debug_datum._shapes_list[-1][-1]
            if(data_layout == "NCHW"):
                config["output shape"] = [None, out_shape[2], out_shape[3], out_shape[1]]
            if(data_layout == "NHWC"):
                config["output shape"] = [None, out_shape[1], out_shape[2], out_shape[3]]

        try:
            t_start  = time.monotonic()
            times = debug_g_mod.run_individual(10, 3, 1000)
            t_end = time.monotonic()
        except:
            print("execution time measurement failed")
            break
            
        layers = {}
        for idx, node in enumerate(debug_g_mod.debug_datum._nodes_list):
            layers[node["op"]] = {
                "time" : float(times[idx])*1000,
            }
            if layer_name in node["op"]:
                layer_time = float(times[idx])*1000
                actual_layer_name = node["op"]
        
        print("layer_time", layer_time)
        runs = int(max(1, np.ceil(time_min_res / (layer_time/1000)))*1.5)

        # determine the noise

        powers = []
        power_normalizer = 1
        if partition != "haswell":
            power_normalizer = 1000
            gpu_utils = []
            mem_utils = []
            gpu_clocks = []
            sm_clocks = []
            mem_clocks = []
            alloc_memory = []
        profile_times = []
                
        #try:
        t_burn_in = 5
        t_start = time.monotonic()
        t_end = t_start + t_burn_in
        while time.monotonic() < t_end:
            # run debug runtime without profiling as burn in
            with suppress_stdout():
                test_data = debug_g_mod.profile(collectors=[], runs=runs, data=tvm.nd.array(inp_data.astype("float32")))

        p_start = time.monotonic()
        for r in range(0, iterations):        
            # reload the Metric Collector due to issues with the PAPI backend
            data_collector = get_data_collector(dev, metrics)
            
            # run debug runtime with time measurements only
            with suppress_stdout():
                test_data = debug_g_mod.profile(collectors=[data_collector], runs=runs, data=tvm.nd.array(inp_data.astype("float32")))
            
            if partition != "haswell":
                pstate = nv.nvmlDeviceGetPowerState(handle)
            #print("\r",(r+1),"PState:", pstate, end="")

            # extract measurement of current run
            if partition == "haswell":
                if dev_idx == 0:
                    relevant = ["CPU0", "DDR_AB", "DDR_CD"]
                elif dev_idx == 1:
                    relevant = ["CPU1", "DDR_EF", "DDR_GH"]
                
                
                for call in test_data.calls:
                    power = 0
                    for value in relevant:
                        #print(value)
                        power += call[value].value
                    if "power" in layers[call["Name"]].keys():
                        layers[call["Name"]]["power"] = max(power, layers[call["Name"]]["power"])
                    else:
                        layers[call["Name"]]["power"] = power

                    if call["Name"] == actual_layer_name:
                        powers.append(power)
                        profile_times.append(call["Duration (us)"].microseconds/1000000/runs)

                    for value in ["CPU0", "DDR_AB", "DDR_CD", "CPU1", "DDR_EF", "DDR_GH", "BLADE"]:
                        if value in layers[call["Name"]].keys():
                            layers[call["Name"]][value] = max(call[value].value, layers[call["Name"]][value])
                        else:
                            layers[call["Name"]][value] = call[value].value
            
            if partition != "haswell":
                for call in test_data.calls:
                    for metric in metrics:
                        if metric in layers[call["Name"]].keys():
                            layers[call["Name"]][metric] = max(layers[call["Name"]][metric], call[metric].value)
                        else:
                            layers[call["Name"]][metric] = call[metric].value
                if call["Name"] == actual_layer_name:
                    powers.append(test_data.calls[0][metrics[0]].value)
                    gpu_utils.append(test_data.calls[0][metrics[1]].value)
                    mem_utils.append(test_data.calls[0][metrics[2]].value)
                    gpu_clocks.append(test_data.calls[0][metrics[3]].value)
                    sm_clocks.append(test_data.calls[0][metrics[4]].value)
                    mem_clocks.append(test_data.calls[0][metrics[5]].value)
                    alloc_memory.append(test_data.calls[0][metrics[6]].value)
                    profile_times.append(test_data.calls[0]["Duration (us)"].microseconds/1000000/runs) # in seconds
            #time.sleep(1)

        p_delta = time.monotonic() - p_start
        avg_power = np.mean(powers)/power_normalizer
        max_power = np.max(powers)/power_normalizer
        min_power = np.min(powers)/power_normalizer
        std_power = np.std(powers)/power_normalizer
        #calculate Z-Score
        z_scores = ((np.array(powers)/power_normalizer) - avg_power)/std_power
        cleaned_powers = []

        if partition != "haswell":
            threshold = 0.25
            attempts = 0
            while len(cleaned_powers) < 3 and attempts < 7:
                attempts += 1
                cleaned_powers = []
                threshold += 0.05
                for idx, score in enumerate(z_scores):
                    if abs(score) < threshold:
                        cleaned_powers.append(powers[idx]/power_normalizer)
            
            if len(cleaned_powers) == 0:
                layer_power = np.max(powers)/power_normalizer
            else:
                layer_power = np.median(cleaned_powers)
        else:
            layer_power = np.max(powers)/power_normalizer
        
        if partition != "haswell":
            layer_memory = np.median(alloc_memory)/(1024**3)
        else:
            layer_memory = -1

        #print()
        print(layers)
        print("raw readings:")
        print(powers)
        print()
        print("outlier removed powers:")
        print(cleaned_powers)
        #print("threshold:")
        #print(threshold)
        print()
        print("final label:")
        print(layer_power)
        #input("c")

        measurements[name+"_"+str(batch_size)] = (layer_time, layer_power, layer_memory)
        config["layers"] = layers
        print("layers:", len(test_data.calls), len(layers))
        print(name, batch_size, (layer_time, layer_power, layer_memory))
        #input("c?")

        '''except:
            print("failed power measurement")
            #exit()
            break'''
            
        if "pool" in workload or "conv" in workload:
            config["data layout"] = data_layout
        if workload == "conv2d":
            config["kernel layout"] = kernel_layout
        config["batch_size"] = batch_size
        config["time"] = layer_time
        config["power"] = layer_power
        config["memory"] = layer_memory
        config["relay"] = str(expr)

        json_text = json.dumps(config)
        with open(file, "w") as f:
            f.write(json_text)
            print("profiling run completed")
        #exit()

print("profiling done")
            
