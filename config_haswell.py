import tvm
from components import profiling

# workload config
repeat = 1024
data_layout = "NCHW"
kernel_layout = "OIHW"
batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
dense_extension = [2, 3]

# defining important variables for the profiling system

#target and device config
target = "llvm -mcpu=haswell"
target_class = "llvm -mcpu=haswell"
device = "E5-2680"
dev_idx = 0
dev = tvm.device(str("cpu"), dev_idx)

# profiling metrics

metrics = [
    "DDR_AB",
    "DDR_CD",
    "DDR_EF",
    "DDR_GH",
    "CPU_0",
    "CPU_1",
    "BLADE"
]

#builder for data_collector
def get_data_collector(dev, metrics, component="hdeem"):
    collector = tvm.runtime.profiling.HDEEMMetricCollector()
    return collector

# sampling resolution
time_min_res = 0.3
iterations = 50

### currently unused
state_path = "./states"
state_file = "state"