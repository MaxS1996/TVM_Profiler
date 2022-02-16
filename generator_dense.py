#"xception:predictions": {"units": 1000, "input shape": [null, 2048], "output shape": [null, 1000], "output dtype": "float32", "compute dtype": "float32"}
import json
import numpy as np
import random

sample_count = 5000
unit_range = list(range(1, 5000, 1))
feature_range = list(range(1, 25000, 1))
output_dtype = "float32"
compute_dtype = "float32"

samples = {}
for index in range(0, sample_count, 1):
    sample = {}
    sample["units"] = random.sample(unit_range, 1)[0]
    sample["features"] = random.sample(feature_range, 1)[0]

    sample["input shape"] = [None, sample["features"]]
    sample["output shape"] = [None, sample["units"]]

    sample["output dtype"] = output_dtype
    sample["compute dtype"] = compute_dtype
    sample["workload"] = "dense"

    samples["random_dense_"+str(index)] = sample
    print("random_dense_"+str(index), ":", sample)

print("generation done")

json_txt = json.dumps(samples)
with open("rand_dense.json", "w") as file:
    file.write(json_txt)
print("data has been written to disk")
print()

