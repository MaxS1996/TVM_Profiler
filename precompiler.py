import json
import os
import numpy as np
import tvm
import pickle
from tvm.relay.op.nn.nn import conv2d, dense, max_pool2d, avg_pool2d
import sys, getopt
import argparse
#from tensorflow import keras

workload_options = [
    "conv2d",
    "dilated_conv2d",
    "depthwise_conv2d",
    "pool2d",
    "dense",
    ]

supported_targets = [
    "alpha",
    "haswell",
    "gpu2",
    "rasp4b",
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

target = partition
data_layout = "NCHW"
kernel_layout = "OIHW"
storage_location = "./precompiled"
batch_sizes = [1, 8, 16, 32, 64]
rand_repeat = 1024

target_class = tvm.target.arm_cpu(target)

samples_base_path = "./configs"
if workload == "conv2d":
    dataset_paths = ["conv_layer_config_clean.json"]
    layer_name = "conv2d"
elif workload == "depthwise_conv2d":
    dataset_paths = ["depthwise_conv_layer_config_clean.json"]
    layer_name = "conv2d"
elif workload == "dilated_conv2d":
    dataset_paths = [
        "dilated_2_conv_layer_config_clean.json",
        "dilated_4_conv_layer_config_clean.json",
        "dilated_8_conv_layer_config_clean.json",
        ]
    layer_name = "conv2d"
elif workload == "dense":
    dataset_paths = [
        "dense_layer_config_clean.json",
        "rand_dense.json",
    ]
    layer_name = "dense"
elif workload == "pool2d":
    dataset_paths = [
        "avg_pool_layer_config_clean.json",
        "max_pool_layer_config_clean.json",
        "rand_pool.json",
        ]

dataset = {}
for dataset_path in dataset_paths:
    with open(samples_base_path+"/"+dataset_path, "r") as file:
        raw = file.read()
        dataset.update(json.loads(raw))

print("import complete")

for name, config in dataset.items():
    print(name)

    config["data_layout"] = data_layout
    config["kernel_layout"] = kernel_layout
    config["workload"] = workload
    
    for batch_size in batch_sizes:
        path = storage_location + "/" + target + "/" + workload + "/"+ name.replace(":", "-").replace("/", "-").replace(".json", "") + "_" + str(batch_size)

        if os.path.exists(path):
            print("already processed.. skipping")
            continue
        config["batch_size"] = batch_size

        if workload != "dense":
            #input shape is always NHWC in json file
            config["N_I"] = batch_size
            config["H_I"] = config["input shape"][1]
            config["W_I"] = config["input shape"][2]
            config["C_I"] = config["input shape"][3]

            config["N_O"] = batch_size
            config["H_O"] = config["output shape"][1]
            config["W_O"] = config["output shape"][2]
            config["C_O"] = config["output shape"][3]


            if data_layout == "NCHW":
                input_shape = (
                    config["N_I"],
                    config["C_I"],
                    config["H_I"],
                    config["W_I"],
                )
            elif data_layout == "NHWC":
                input_shape = (
                    config["N_I"],
                    config["H_I"],
                    config["W_I"],
                    config["C_I"],
                )
            else:
                raise NotImplementedError()
        elif workload == "dense":
            input_shape = (
                int(batch_size),
                int(config["input shape"][1]),
            )

        required = int(np.prod(input_shape))
        raw_input_data = np.random.rand(int(np.ceil(required/rand_repeat)))
        input_data = np.repeat(raw_input_data, rand_repeat)[:required].reshape(input_shape).astype("float32")

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
            weight_data = np.random.rand(np.prod(weight_shape)).reshape(weight_shape).astype("float32")
            y = tvm.relay.Constant(tvm.nd.array(weight_data))

        x = tvm.relay.var("data", tvm.relay.TensorType(input_shape), dtype="float32")

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

        metas = dict(compiled_graph_lib.function_metadata)
        layer_name = list(metas.keys())[0]
        config["io_size"] = list(dict(metas[layer_name].io_sizes).values())[0].value
        config["ws_size"] = list(dict(metas[layer_name].workspace_sizes).values())[0].value

        #store compiled
        
        folders = path.split("/")

        tmp = ""
        for folder in folders:
            tmp = tmp + folder + "/"
            if not os.path.exists(tmp):
                os.mkdir(tmp)
        del tmp

        #write config
        raw_config = json.dumps(config)
        with open(path+"/config.json", "w") as  file:
            file.write(raw_config)
        
        #export library
        compiled_graph_lib.export_library(path+"/lib.tar")

        #export graph
        with open(path+"/graph.json", "w") as file:
            file.write(compiled_graph_lib.graph_json)

        with open(path + "/compressed_input.data", "wb") as file:
            pickle.dump([raw_input_data, input_shape], file)

        #export params
        with open(path+"/data.params", "wb") as file:
            pickle.dump(params, file)
        print(config)
        print("  compiled")

print("done")