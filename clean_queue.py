import json
import numpy as np
import pandas as pd

workloads = ["max_pool2d", "avg_pool2d", "dense", "conv2d"]


for workload in workloads:
    print("cleaning up "+workload+" sample set")
    if workload == "conv2d":
        workload_path = "conv_layer_config.json"
    elif workload == "dense":
        workload_path = "dense_layer_config.json"
    elif workload == "avg_pool2d":
        workload_path = "avg_pool_layer_config.json"
    elif workload == "max_pool2d":
        workload_path = "max_pool_layer_config.json"

    list_features = {
            "input shape" : ["N_I", "H_I", "W_I", "C_I"],
            "output shape" : ["N_O", "H_O", "W_O", "C_O"],
        }
    features = ["output dtype", "compute dtype"]

    if workload in ["avg_pool2d", "max_pool2d", "conv2d"]:
        features += ["padding"]
        list_features.update({
            "strides" : ["strides_0", "strides_1"],
        })

    if "pool" in workload:
        list_features.update({
            "pool_size" : ["pool_0", "pool_1"],
        })

    if "dense" in workload:
        list_features.update({
            "input shape" : ["N_I", "H_I"],
            "output shape" : ["N_O", "H_O"],
        })

    if "conv2d" == workload:
        features += ["filters", "groups"]
        list_features.update({
            "kernel" : ["kernel_0", "kernel_1"],
            "dilation" : ["dilation_0", "dilation_1"],
        })

    with open(workload_path, "r") as file:
        configs = json.load(file)
    print("json file has been read")

    df = pd.DataFrame.from_dict(configs, orient='index')
    print("dataframe generated")
    print("full workset size", len(df))
    print("expand list features")
    for name, heads in list_features.items():
        print(name, "\t:\t", heads)
        split_df = pd.DataFrame(df[name].tolist(), columns=heads)
        split_df.index = df.index
        df = pd.concat([df, split_df], axis=1)
        features += heads

    df = df.drop_duplicates(subset=features)
    print("after duplicate removal", len(df))
    for name, heads in list_features.items():
        df = df.drop(heads, axis=1)


    df = df.T
    print("writting back to JSON")
    target_path = workload_path.replace(".json", "_clean.json")
    raw_json = df.to_json()
    with open(target_path, "w") as file:
        file.write(raw_json)
    print(workload + " done...")
    print()
print("done")

