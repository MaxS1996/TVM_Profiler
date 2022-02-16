'''
"inception_v3:average_pooling2d":
    {
        "pool_size": [3, 3],
        "strides": [1, 1],
        "padding": "same",
        "input shape": [null, 35, 35, 192],
        "output shape": [null, 35, 35, 192],
        "output dtype": "float32",
        "compute dtype": "float32"
    }
'''
import json
import numpy as np
import random
from copy import deepcopy

sample_count = 1000
pool_range = [2, 4, 6]
stride_range = [1, 2, 3, 4]

w_i_range = list(range(7, 200, 1))
c_i_range = list(range(40, 4096, 1))
pad_range = ["same", "valid"]

output_dtype = "float32"
compute_dtype = "float32"

samples = {}
for index in range(0, sample_count, 1):
    sample = {}

    # pool size
    pool = random.sample(pool_range, 1)[0]
    sample["pool_size"] = [pool, pool]

    # stride
    stride = random.sample(pool_range, 1)[0]
    sample["stride"] = [stride, stride]

    # padding
    sample["padding"] = random.sample(pad_range, 1)[0]

    # N C H W
    height = random.sample(w_i_range, 1)[0]
    width = random.sample(w_i_range, 1)[0]
    channels = random.sample(c_i_range, 1)[0]
    sample["input shape"] = [None, height, width, channels]

    new_height = (height - pool)//stride + 1
    new_width = (width - pool)//stride + 1

    sample["output shape"] = [None, new_height, new_width, channels]

    sample["output dtype"] = output_dtype
    sample["compute dtype"] = compute_dtype
    sample["workload"] = "avg_pool2d"

    samples["random_avg_pool_"+str(index)] = sample
    print("random_avg_pool_"+str(index), ":", sample)

    max_sample = deepcopy(sample)
    sample["workload"] = "max_pool2d"
    samples["random_max_pool_"+str(index)] = max_sample
    print("random_max_pool_"+str(index), ":", sample)

print("generation done")

json_txt = json.dumps(samples)
with open("rand_pool.json", "w") as file:
    file.write(json_txt)
print("data has been written to disk")
print()

