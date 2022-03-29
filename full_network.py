# IMPORTS from tvm, different required packages and the profiling infrastructure

from copy import deepcopy
import tvm
import tvm.relay as relay
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.debugger import debug_executor as graph_runtime
#####
import numpy as np
#import pynvml as nv
from func_timeout import func_timeout
import time
import json
import sys
import os

# helpful to suppress output of debug runtime run function
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

os.environ["TVM_BACKTRACE"] = "1"
#####
from components import description_vector as dv
from components import serializer
from components import profiling

from tensorflow import keras as keras
from keras.models import Sequential, Functional
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D

#####################################################
#dataset_path = "/home/s0144002/DIR/ssd/s0144002-TVMMapper/TVM_Profiling_Dataset/dataset"
dataset_path = "./testing/test"
#####################################################

import sys, getopt
import argparse
workload = "alexnet"
workload_options = [
    "alexnet",
    "darknet",
    "mnist_net",
    ]

supported_targets = [
    "alpha",
    "haswell",
    "gpu2",
    "980ti",
    "test",
]

iterations = 10

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument(
        "-t",
        "--target",
        default="alpha",
        help="The target device, you want to compile for and profile on.")
    parser.add_argument(
        "-w",
        "--workload",
        default="darknet",
        help="The network")
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

#####################################################

if partition == "alpha":
    from config_alpha import *

elif partition == "haswell":
    from config_haswell import *

elif partition == "gpu2":
    from config_gpu2 import *

elif partition == "980ti":
    from config_980ti import *

elif partition == "test":
    from config_test import *

#####################################################

if workload == "alexnet":
    from mxnet.gluon.model_zoo import vision
    alexnet = vision.alexnet(pretrained=True)

    model_name = "alexnet"
    model_source = "mxnet"
    input_name = "data"
    input_shape = [3,224,224]

if workload == "mnist_net":
    model_name = "mnist_net"
    model_source = "keras"
    input_name = "input_1"
    input_shape = [1,28,28]

    img_inputs = keras.Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(img_inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=[img_inputs], outputs=x, name="mnist_model")

if workload == "darknet":
    model_name = "darknet-19"
    model_source = "keras"
    input_name = "input_1"
    input_shape = [3,224,224]
    
    img_inputs = keras.Input(shape=(224, 224, 3))
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', input_shape=(224,224,3))(img_inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(filters=1000, kernel_size=(1,1), strides=(1,1), padding='valid')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = keras.Model(inputs=[img_inputs], outputs=x, name="darknet")

#model.summary()


#################### Network has been converted - Next Step: Compile
for batch_size in batch_sizes:
    file = dataset_path+"/"+target_class+"_"+device+"/test_set/"+model_name.replace(":", "-").replace("/","_")+"_"+str(batch_size)+".json"

    folders = file.split("/")[0:-1]
    tmp = folders[0]
    
    for folder in folders[1::]:
        tmp = tmp + "/" + folder
        if not os.path.exists(tmp):
            os.mkdir(tmp)

    bn_input_shape = [batch_size] + input_shape
    shape_dict = {input_name: tuple(bn_input_shape)}
    if model_source == "mxnet":
        mod, params = relay.frontend.from_mxnet(alexnet, shape_dict)
    elif model_source == "keras":
        mod, params = relay.frontend.from_keras(model, shape_dict)
    print(model_name, "converted to Relay")

    with tvm.transform.PassContext(opt_level=3):
        print(target_class)
        compiled_graph_lib = tvm.relay.build_module.build(mod, target_class, params=params)
        #compiled_graph_lib = relay.build_module.create_executor("graph", mod, dev, target_class, params)

    print("compiled the model")

    debug_g_mod = graph_runtime.GraphModuleDebug(
        compiled_graph_lib["debug_create"]("default", dev),
        [dev],
        compiled_graph_lib.get_graph_json(),
        "."
    )

    t_start  = time.monotonic()
    times = debug_g_mod.run_individual(10, 3, 1000)
    t_end = time.monotonic()
    print("time-only layer-wise measurement took:",t_end - t_start)
    print()

    layers = {}
    layer_order = []
    for idx, node in enumerate(debug_g_mod.debug_datum._nodes_list):
        
        func_name = "param"
        hash_val = ""
        if node["op"] != "param":
            func_name = node["attrs"]["func_name"]
            hash_val = node["attrs"]["hash"]
        layers[node["name"]] = {
            "time" : float(times[idx])*1000,
            "func_name" : func_name,
            "hash" : hash_val
        }
        layer_order.append(node["name"])
    print("measured", len(layers), "individual layers")

    print("end2end time measurement")
    full_time = debug_g_mod.benchmark(dev, repeat=10, number=3, end_to_end=True).median
    print("TIME MEASUREMENT COMPLETED")

    min_layer_time = float("inf")
    min_time_name = ""
    for layer, measured in layers.items():
        if measured["time"] < min_layer_time and measured["func_name"] != "param":
            min_layer_time = measured["time"]
            min_time_name = layer

    print("min layer time:", min_time_name, "with:", min_layer_time)
    runs = int(max(1, np.ceil(time_min_res / (min_layer_time/1000))))
    print("required repitions to get acceptable power measurement precision:", runs)

    required = int(np.prod(bn_input_shape))
    rand_data = np.random.rand(int(np.ceil(required/repeat)))
    inp_data = np.repeat(rand_data, repeat)[:required].reshape(bn_input_shape).astype("float32")
    inp_dict = {input_name : tvm.nd.array(inp_data.astype("float32"))}

    t_burn_in = 5
    t_start = time.monotonic()
    t_end = t_start + t_burn_in
    while time.monotonic() < t_end:
        # run debug runtime without profiling as burn in
        with suppress_stdout():
            test_data = debug_g_mod.profile(collectors=[], runs=runs, **inp_dict)

    p_start = time.monotonic()
    for r in range(0, iterations):        
        # reload the Metric Collector due to issues with the PAPI backend
        data_collector = get_data_collector(dev, metrics)
        
        # run debug runtime with time measurements only
        with suppress_stdout():
            test_data = debug_g_mod.profile(collectors=[data_collector], runs=runs, **inp_dict)
            #test_data = debug_g_mod.profile(collectors=[], runs=runs, **inp_dict)

        for call in test_data.calls:
            #print(call)
            #input()
            if not "Percent" in layers[call["Name"]].keys():
                layers[call["Name"]]["Percent"] = []
            layers[call["Name"]]["Percent"].append(call["Percent"].percent)
            for metric in metrics:
                if not metric in layers[call["Name"]].keys():
                    layers[call["Name"]][metric] = []
                layers[call["Name"]][metric].append(call[metric].value)
    
    print(layers)
    print("batch size done")

    collected_data = {
        "name" : model_name,
        "batch_size" : batch_size,
        "target" : target,
        "device" : device,
        "data" : layers,
        "order" : layer_order,
        "total_time" : full_time,
    }

    json_text = json.dumps(collected_data)
    with open(file, "w") as f:
        f.write(json_text)
    #exit()