import enum
import numpy as np
import tvm
from tvm.relay.analysis import analysis as analysis

class TIRType (enum.Enum):
    TIR_INPUT     = 0
    TIR_PARAM     = 1
    TIR_OPERATION = 2
    TIR_UNKNOWN   = 3
    
class TIRFunction():
    name = ""
    shape = []
    dtype = ""
    op_type = TIRType.TIR_UNKNOWN

    time = 0.0
    inputs = {} #input object + reuse factor & access pattern
    outputs = {} #output object + reuse factor & access pattern

    ratios = {
        "input-output": 1.0,
        "input-weight": 1.0,
        "weight-output": 1.0,
    }

    flops = 0
    iops = 0
    mem = 0

    def __init__(self, node, time, inputs=[], outputs=[], is_input=False):
        self.name = node["name"]
        self.shape = node["shape"]
        self.dtype = node["attrs"]["T"].split(": ")[-1]
        self.time = time

        if node["op"] == "param" and not is_input:
            self.op_type = TIRType.TIR_PARAM
        else:
            self.op_type = TIRType.TIR_OPERATION
        
        if is_input:
            self.op_type = TIRType.TIR_INPUT
        
        return

def get_ops(func):
    if(not type(func) is tvm.relay.function.Function):
        return []

    ops = list()

    args = list()
    args.append(func.body)
    while len(args) != 0:
        node = args[0]
        args.pop(0)

        if type(node) == tvm.relay.expr.Tuple:
            args += list(node.fields)
        else:
            if not type(node) == tvm.relay.expr.Var and not type(node) == tvm.relay.expr.Constant:
                args += list(node.args)
                ops.append((node.op.name, node))
        
        #print(analysis.get_total_mac_number(node))
        
    return ops

def get_dominant_ops(func):
    ops = get_ops(func)
    dominant = list()

    for op in ops:
        if "conv" in op[0]:
            dominant.append(op)
        if "pool" in op[0]:
            dominant.append(op)
        if "dense" in op[0]:
            dominant.append(op)

    return dominant

def get_description_vector(func):
    ops = get_dominant_ops(func)
    dom = list()
    for op in ops:
        desc = dict()

        if "conv" in op[1].op.name:
            # calculation info
            desc["op"] = "conv"
            desc["dtype"] = op[1].checked_type.dtype
            desc["mac"] = analysis.get_total_mac_number(op[1])
            
            if(type(op[1].attrs) == tvm.relay.op.op_attrs.ConvWinogradWeightTransformAttrs):
                desc["kernel_size"] = [op[1].attrs.tile_size, op[1].attrs.tile_size]
                desc["padding"] = [0, 0, 0, 0]
                desc["dilation"] = [0, 0]
                desc["strides"] = [0, 0]
                desc["groups"] = 0
                desc["channels"] = 0
            else:
                desc["kernel_size"] = list(op[1].attrs.kernel_size)
                desc["padding"] = list(op[1].attrs.padding)
                desc["dilation"] = list(op[1].attrs.dilation)
                desc["strides"] = list(op[1].attrs.strides)
                desc["groups"] = op[1].attrs.groups
                desc["channels"] = op[1].attrs.channels
            desc["pool_size"] = [0, 0]

        if "dense" in op[1].op.name:
            # calculation info
            desc["op"] = "dense"
            desc["dtype"] = op[1].checked_type.dtype
            desc["mac"] = analysis.get_total_mac_number(op[1])
            desc["kernel_size"] = [0, 0]
            desc["pool_size"] = [0, 0]
            desc["padding"] = [0, 0, 0, 0]
            desc["dilation"] = [0, 0]
            desc["strides"] = [0, 0]
            desc["groups"] = 0
            desc["channels"] = 0

        if "pool" in op[1].op.name:
            if(type(op[1].attrs) == tvm.relay.op.op_attrs.AdaptivePool2DAttrs):
                #print("break, not good")
                desc["op"] = "adaptive_max_pool"
                desc["pool_size"] = [0, 0]
                desc["padding"] = [0, 0]
                desc["strides"] = [0, 0]
            else:
                # calculation info
                if "max" in op[1].op.name:
                    desc["op"] = "max_pool"
                if "avg" in op[1].op.name:
                    desc["op"] = "avg_pool"
                desc["dtype"] = op[1].checked_type.dtype
                desc["mac"] = analysis.get_total_mac_number(op[1])
                desc["kernel_size"] = [0, 0]
                if not "global" in op[1].op.name:
                    desc["pool_size"] = list(op[1].attrs.pool_size)
                    desc["padding"] = list(op[1].attrs.padding)
                    desc["strides"] = list(op[1].attrs.strides)
                else:
                    print("global")
                    desc["op"] += "_global"
                    desc["pool_size"] = [op[1].args[0].checked_type.concrete_shape[2], op[1].args[0].checked_type.concrete_shape[3]]
                    desc["padding"] = [0, 0, 0, 0]
                    desc["strides"] = [0, 0]
            
            desc["dilation"] = [0, 0]
            desc["groups"] = 0
            desc["channels"] = 0

        #input tensor
        desc["input_shape"] = op[1].args[0].checked_type.concrete_shape
        desc["input_type"] = op[1].args[0].checked_type.dtype
        if "conv" in op[1].op.name:
            if(type(op[1].attrs) == tvm.relay.op.op_attrs.ConvWinogradWeightTransformAttrs):
                desc["input_layout"] = ""
            else:    
                desc["input_layout"] = op[1].attrs.data_layout
        if "pool" in op[1].op.name:
            desc["input_layout"] = op[1].attrs.layout
        if "dense" in op[1].op.name:
            desc["input_layout"] = ""

        #weight tensor
        if not "pool" in op[1].op.name:
            if(type(op[1].attrs) == tvm.relay.op.op_attrs.ConvWinogradWeightTransformAttrs):
                desc["weight_shape"] = ""
                desc["weight_type"] = ""
            else:
                desc["weight_shape"] = op[1].args[1].checked_type.concrete_shape
                desc["weight_type"] = op[1].args[1].checked_type.dtype
            if "conv" in op[1].op.name:
                if(type(op[1].attrs) == tvm.relay.op.op_attrs.ConvWinogradWeightTransformAttrs):
                    desc["weight_layout"] = ""
                else:
                    desc["weight_layout"] = op[1].attrs.kernel_layout
            else:
                desc["weight_layout"] = ""
        else:
            desc["weight_shape"] = [0, 0, 0, 0, 0]
            desc["weight_type"] = ""
            desc["weight_layout"] = ""

        #out tensor
        desc["output_shape"] = op[1].checked_type.concrete_shape
        desc["output_type"] = op[1].checked_type.dtype
        if "conv" in op[1].op.name:
            if(type(op[1].attrs) == tvm.relay.op.op_attrs.ConvWinogradWeightTransformAttrs):
                desc["output_layout"] = ""
            else:
                desc["output_layout"] = op[1].attrs.out_layout
        if "pool" in op[1].op.name:
            desc["output_layout"] = op[1].attrs.layout
        if "dense" in op[1].op.name:
            desc["output_layout"] = ""

        dom.append(desc)

    return dom

