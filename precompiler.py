import json
import os
import numpy as np
import tvm
import pickle
from tvm.relay.op.nn.nn import conv2d, dense, max_pool2d, avg_pool2d
#from tensorflow import keras

target = "rasp4b"
data_layout = "NCHW"
kernel_layout = "OIHW"
workload = "conv2d"
storage_location = "./precompiled"
batch_sizes = [1, 8, 16, 32, 64]
rand_repeat = 1024

target_class = tvm.target.arm_cpu(target)

if workload == "conv2d":
    dataset_path = "conv_layer_config.json"

with open(dataset_path, "r") as file:
    raw = file.read()
    dataset = json.loads(raw)

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

        required = int(np.prod(input_shape))
        raw_input_data = np.random.rand(int(np.ceil(required/rand_repeat)))
        input_data = np.repeat(raw_input_data, rand_repeat)[:required].reshape(input_shape).astype("float32")

        if kernel_layout == "OIHW":
            weight_shape = (
                config["C_O"],
                config["C_I"],
                config["kernel"][0],
                config["kernel"][1]
            )
        else:
            raise NotImplementedError()

        weight_data = np.random.rand(np.prod(weight_shape)).reshape(weight_shape).astype("float32")

        x = tvm.relay.var("data", tvm.relay.TensorType(input_shape), dtype="float32")
        y = tvm.relay.Constant(tvm.nd.array(weight_data))

        if workload == "conv2d":
            expr = conv2d(
                data = x,
                weight= y,
                strides=config["strides"],
                padding=0,
                dilation=config["dilation"],
                groups=config["groups"],
                channels=int(config["filters"]),
                kernel_size=config["kernel"],
                data_layout=data_layout,
                kernel_layout=kernel_layout,
                out_dtype=config["output dtype"],
                )
        else:
            raise NotImplementedError()

        mod = tvm.ir.IRModule.from_expr(expr)
        params = {}

        with tvm.transform.PassContext(opt_level=3):
            compiled_graph_lib = tvm.relay.build_module.build(mod, target_class, params=params)

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