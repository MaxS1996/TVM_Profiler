import tvm
from components import profiling

# workload config
repeat = 1024
data_layout = "NCHW"
kernel_layout = "OIHW"
batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]
dense_extension = [2,3]

# defining important variables for the profiling system

#target and device config
target = "cuda"
target_class = "cuda"
device = "K80"
dev_idx = 0
dev = tvm.device(str("cuda"), dev_idx)

# profiling metrics

papi_base = "nvml:::Tesla_K80:device_"
metrics = []
if device == "K80":
    metrics = profiling.get_metrics(target, device, backend="nvml", dev_idx=dev_idx)
    metrics.append(papi_base+str(dev_idx)+":gpu_utilization")
    metrics.append(papi_base+str(dev_idx)+":memory_utilization")
    metrics.append(papi_base+str(dev_idx)+":graphics_clock")
    metrics.append(papi_base+str(dev_idx)+":sm_clock")
    metrics.append(papi_base+str(dev_idx)+":memory_clock")
    metrics.append(papi_base+str(dev_idx)+":allocated_memory")

#builder for data_collector
def get_data_collector(dev, metrics, component="nvml"):
    collector = tvm.runtime.profiling.PAPIMetricCollector({dev: metrics}, component=component)
    return collector

# sampling resolution
time_min_res = 0.2

### currently unused
state_path = "./states"
state_file = "state"
