import sys, getopt
import argparse

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
    "gpu2"
]

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