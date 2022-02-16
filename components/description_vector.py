import enum
import numpy as np
import tvm
from tvm.relay.analysis import analysis as analysis
import pickle
import os

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

def get_size(checked_type):
    size = 1
    for s in checked_type.concrete_shape:
        size *= s

    if checked_type.dtype == "float32":
        size *= 4
    
    if checked_type.dtype == "int32":
        size *= 4

    if checked_type.dtype == "float16":
        size *= 2
    
    return size

def fix_data(x, length = 4):
    if (not type(x) is tuple) and (not type(x) is list):
        y = x
        x = list()
        x.append(y)

    if type(x) is tuple:
        x = list(x)
    if type(x) is list:
        while len(x) < length:
            x.append(0)
    return x

#does ONLY handle standard layouts
def get_axes(shape, axis="C", layout="NCHW"):
    dims = list(layout)
    for d in range(len(dims)):
        if dims[d] == axis:
            return shape[d]
    
    return None


def from_metainfo(meta):
    func = meta.relay_primfuncs.items()[0][1]

    debug_data = None
    op_desc = OperationDescription.from_relay_func(func)

    op_config = OperatorConfig.from_relay_func(func)

    input_config = TensorConfig.from_relay_func(func, "input")
    output_config = TensorConfig.from_relay_func(func, "output")
    if "conv2d" in op_desc.dominant_op or "dense" in op_desc.dominant_op:
        weight_config = TensorConfig.from_relay_func(func, "weight")
    else:
        weight_config = None

    collector_info = None

    return DescriptionVector(debug_data, op_desc, op_config, input_config, weight_config, output_config, collector_info)
    
def from_pickle(path):
    if os.path.exists(path) and os.path.isfile(path):
        with open(path, "rb") as f:
            rv = pickle.load(f)
        return rv
    return None

def from_folder(path):
    rv = {}

    content = os.listdir(path)
    while(len(content) != 0):
        entry = path+"/"+content[-1]
        if os.path.isdir(entry):
            rv.update(from_folder(entry))
        if os.path.isfile(entry):
            if "1.total" in entry:
                rv.update(from_pickle(entry))
        del content[-1]      

    return rv


class DebugInfo():
    head = [
        "Layer Name",
        "Network Name"
    ]
    def __init__(self, name, network):
        self.name = name
        self.network = network
        return

    def get_scheme(self):
        values = dict()
        if self != None:
            values["Layer Name"] = self.name
            values["Network Name"] = self.network
        else:
            values["Layer Name"] = ""
            values["Network Name"] = ""

        return values

    def to_txt(self):
        txt = ""
        txt += "name: "+ str(self.name) + "\n"
        txt += "network: "+ str(self.network) + "\n"
        return txt

    #TODO: create from layer/ database entry/ ...

    def for_csv(self):
        values = [self.name, self.network]
        return values

class OperationDescription():
    head = [
        "Dominant Operation",
        "Operation Datatype",
        "#MAC",
        "#MUL",
        "#ADD"
    ]
    
    def __init__(self, dom_op, op_type, mac, mul, add):
        self.dominant_op = dom_op
        self.op_type = op_type
        self.mac_count = mac
        self.mul_count = mul
        self.add_count = add
        return

    def get_scheme(self):
        values = dict()
        if self != None:
            values["Dominant Operation"] = self.dominant_op
            values["Operation Datatype"] = self.op_type
            values["#MAC"] = self.mac_count
            values["#MUL"] = self.mul_count
            values["#ADD"] = self.add_count
        else:
            values["Dominant Operation"] = ""
            values["Operation Datatype"] = ""
            values["#MAC"] = 0
            values["#MUL"] = 0
            values["#ADD"] = 0

        return values

    def from_relay_func(func):
        dom_op = get_dominant_ops(func)
        
        if len(dom_op) > 1:
            print("#TODO: handle fused functions with mutliple dominant ops")
        
        if len(dom_op) != 0:
            node = dom_op[0][1]
            if "conv2d" in dom_op[0][0]:
                op = "conv2d"

            if "dense" in dom_op[0][0]:
                op = "dense"

            if "pool" in dom_op[0][0]:
                op = "pool"

                if "max" in dom_op[0][0]:
                    op = "max_"+op
                if "avg" in dom_op[0][0]:
                    op = "avg_"+op
                if "global" in dom_op[0][0]:
                    op = "global_"+op    
        else:
            node = get_ops(func)[0][1]
            op = "other"
        op_type = node.checked_type.dtype
        
        mac_count = analysis.get_total_mac_number(node)
        if mac_count == 0:
            mac_count = OperationDescription.calc_mac(node, op)
        ops = OperationDescription.calc_ops(node, op)
        mul_count = ops[0]
        add_count = ops[1]

        return OperationDescription(op, op_type, mac_count, mul_count, add_count)

    def calc_mac(node, op):
        mac = 0
        in_shape = node.args[0].checked_type.concrete_shape
        in_prod = float(np.product(np.ma.masked_equal(in_shape, 0)))

        if "dense" in op:
            mac = in_prod * node.args[1].checked_type.concrete_shape[0]

        return mac


    def calc_ops(node, op):
        #if len(get_ops(node)) == 0:
        #    return (0,0)

        if type(node.args[0].checked_type) == tvm.ir.type.TupleType:
            #print("stop")
            return (0,0)

        in_shape = node.args[0].checked_type.concrete_shape
        in_elems = float(np.product(np.ma.masked_equal(in_shape, 0)))

        out_shape = node.checked_type.concrete_shape
        out_elems = float(np.product(np.ma.masked_equal(out_shape, 0)))

        if "conv2d" in op:
            weight_shape = node.args[1].checked_type.concrete_shape
            weight_elems = float(np.product(np.ma.masked_equal(weight_shape, 0)))
            channels = node.attrs.channels.value

            mul =  weight_elems
            mul *= out_elems
            mul /= channels

            add = weight_elems-1
            add *= out_elems
            add /= float(channels)

            return (mul, add)

        if "dense" in op:
            weight_shape = node.args[1].checked_type.concrete_shape
            weight_elems = float(np.product(np.ma.masked_equal(weight_shape, 0)))

            mul = float(out_elems) * weight_shape[-1]

            add = float(out_elems) * (weight_shape[-1]-1)

            return (mul, add)

        if op == "max_pool":
            pool_size = list(node.attrs.pool_size)
            for i in range(len(pool_size)):
                pool_size[i] = pool_size[i].value
            pool_elems = float(np.product(np.ma.masked_equal(pool_size, 0)))
            add = out_elems * (pool_elems -1)

            return (0, add)

        if type(node.attrs) == tvm.relay.op.op_attrs.AdaptivePool2DAttrs:
            print("AdaptivePool2D")
            return (0,0)

        if op == "avg_pool":
            pool_size = list(node.attrs.pool_size)
            for i in range(len(pool_size)):
                pool_size[i] = pool_size[i].value
            pool_elems = float(np.product(np.ma.masked_equal(pool_size, 0)))
            mul = out_elems
            add = out_elems * (pool_elems - 1)

            return (mul, add)

        if "global" in op:
            #channels = get_axes(node.checked_type.concrete_shape, "C", node.attrs.layout)
            if hasattr(node.attrs, "layout"):
                layout = node.attrs.layout
            if hasattr(node.attrs, "dst_layout"):
                layout = node.attrs.dst_layout
            channels = TensorConfig.extract_dims(node.checked_type.concrete_shape, layout)[1]
        if op == "global_max_pool":
            add = out_elems - channels
            return (0, add)

        if op == "global_avg_pool":
            
            add = in_elems / out_elems * channels

            return (channels, add)

        return (0,0)          

    def to_txt(self):
        txt = ""
        txt += "dominant_op: "+str(self.dominant_op) + "\n"
        txt += "op_type: "+str(self.op_type) + "\n"
        txt += "mac_count: "+str(self.mac_count) + "\n"
        txt += "mul_count: "+str(self.mul_count) + "\n"
        txt += "add_count: "+str(self.add_count) + "\n"
        return txt

    def for_csv(self):
        values = [self.dominant_op, self.op_type, self.mac_count, self.mul_count, self.add_count]
        return values
    #TODO: create from layer/ database entry/ ...

class OperatorConfig():

    head = [
        "Kernel Size",
        "Padding",
        "Dilation",
        "Strides",
        "Groups",
        "Channels",
        "Pool Size"
    ]

    def __init__(self, kernel, padding, dilation, strides, groups, channels, pool):
        self.kernel_size = kernel
        self.padding = padding
        self.dilation = dilation
        self.strides = strides
        self.groups = groups
        self.channels = channels
        self.pool = pool

    def get_scheme(self):
        values = dict()
        if self != None:
            values["Kernel Size"] = self.kernel_size
            values["Padding"] = self.padding
            values["Dilation"] = self.dilation
            values["Strides"] = self.strides
            values["Groups"] = self.groups
            values["Channels"] = self.channels
            values["Pool Size"] = self.pool
        else:
            values["Kernel Size"] = [0, 0]
            values["Padding"] = [0,0,0,0]
            values["Dilation"] = [0,0]
            values["Strides"] = [0,0]
            values["Groups"] = 0
            values["Channels"] = 0
            values["Pool Size"] = [0,0]
        return values

    def from_relay_func(func):
        dom_op = get_dominant_ops(func)
        
        if len(dom_op) > 1:
            print("#TODO: handle fused functions with mutliple dominant ops")
        
        if len(dom_op) < 1:
            print("no dominand op in func")
            op = "other"
            return None
        else:
            node = dom_op[0][1]
            if "conv2d" in dom_op[0][0]:
                op = "conv2d"
            if "dense" in dom_op[0][0]:
                op = "dense"
            if "pool" in dom_op[0][0]:
                op = "pool"
                if "max" in dom_op[0][0]:
                    op = "max_"+op
                if "avg" in dom_op[0][0]:
                    op = "avg_"+op
                if "global" in dom_op[0][0]:
                    op = "global_"+op

        if op == "conv2d":
            kernel_size = fix_data(list(node.attrs.kernel_size), length=2)
            cleaned_kernel = []
            for value in kernel_size:
                if hasattr(value, "value"):
                    cleaned_kernel.append(value.value)
                else:
                    cleaned_kernel.append(value)
            padding = fix_data(list(node.attrs.padding), length=4)
            dilation = fix_data(list(node.attrs.dilation), length=2)
            strides = fix_data(list(node.attrs.strides), length=2)
            groups = float(node.attrs.groups)
            channels = float(node.attrs.channels.value)
            pool = [0, 0]

            return OperatorConfig(kernel_size, padding, dilation, strides, groups, channels, pool)
        
        if op == "dense":
            kernel_size = [0, 0]
            padding = [0, 0, 0, 0]
            dilation = [0, 0]
            strides = [1, 1]
            groups = 1
            channels = 0
            pool = [0, 0]

            return OperatorConfig(kernel_size, padding, dilation, strides, groups, channels, pool)

        if "pool" in op:
            if "global" in op:
                kernel_size = [0, 0]
                padding = [0, 0, 0, 0]
                dilation = [0, 0]
                strides = [1, 1]
                groups = 1
                channels = 0
                pool = fix_data([node.args[0].checked_type.concrete_shape[2], node.args[0].checked_type.concrete_shape[3]], length=2)
                #TODO: figure out strides, padding
                return OperatorConfig(kernel_size, padding, dilation, strides, groups, channels, pool)
            else:
                if type(node.attrs) == tvm.relay.op.op_attrs.AdaptivePool2DAttrs:
                    print("adaptive pooling")
                else:
                    kernel_size = [0, 0]
                    padding = fix_data(list(node.attrs.padding), length=4)
                    dilation = [0, 0]
                    strides = fix_data(list(node.attrs.strides), length=2)
                    groups = 1
                    channels = 0
                    pool = fix_data(list(node.attrs.pool_size), length=2)
                    #TODO: figure out strides, padding
                    return OperatorConfig(kernel_size, padding, dilation, strides, groups, channels, pool)

    def to_txt(self):
        txt = ""
        txt += "kernel_size: "+ str(self.kernel_size) + "\n"
        txt += "padding: "+ str(self.padding) + "\n"
        txt += "dilation: "+ str(self.dilation) + "\n"
        txt += "strides: "+ str(self.strides) + "\n"
        txt += "groups: "+ str(self.groups) + "\n"
        txt += "channels: "+ str(self.channels) + "\n"
        txt += "pool_size: "+ str(self.pool) + "\n"
        return txt
    
    def for_csv(self):
        values = [self.kernel_size, self.padding, self.dilation, self.strides, self.groups, self.channels, self.pool]
        return values

    #TODO: create from layer/ database entry/ ...

class TensorConfig():

    head = [
        "N/O",
        "C/I",
        "H",
        "W"
        "Data Type",
        "Data Layout"
    ]

    def __init__(self, N, H, W, C, I, O, dtype, layout):
        if N is not None:
            self.N = N

        if C is not None:
            self.C = C

        if I is not None:
            self.I = I

        if O is not None:
            self.O = O

        self.H = H
        self.W = W
        
        self.type = dtype
        self.layout = layout

    def get_scheme(self, tensor_type="input"):
        values = dict()
        if self != None:
            if tensor_type == "input" or tensor_type == "output":
                values[tensor_type+" N"] = self.N
                values[tensor_type+" C"] = self.C
            else:
                values[tensor_type+" O"] = self.O
                values[tensor_type+" I"] = self.I

            values[tensor_type+" H"] = self.H
            values[tensor_type+" W"] = self.W
            values[tensor_type+" type"] = self.type
            values[tensor_type+" layout"] = self.layout
        else:
            if tensor_type == "input" or tensor_type == "output":
                values[tensor_type+" N"] = 0
                values[tensor_type+" C"] = 0
            else:
                values[tensor_type+" O"] = 0
                values[tensor_type+" I"] = 0

            values[tensor_type+" H"] = 0
            values[tensor_type+" W"] = 0
            values[tensor_type+" type"] = ""
            values[tensor_type+" layout"] = ""

        return values

    def extract_dims(shape, layout):
        if len(shape) == 3 and layout == "":
            layout = "HW"+str(shape[-1])+"h"
        
        if len(shape) == 2 and (layout == "" or layout == "NCHW"):
            layout = "HW"
        if len(shape) == 4 and layout == "":
            layout = "NCHW"
        if len(shape) == 5 and layout == "":
            layout = "NCHW"+str(shape[-1])+"c"
            #TODO: fix layout propagation
        dims = list(layout)
        N, H, W, C, I, O = 0, 0, 0, 0, 0, 0

        it = 0
        digits = ""
        for d in range(len(dims)):
            if dims[d] == "N":
                N = shape[it]
            if dims[d] == "I":
                I = shape[it]
            if dims[d] == "O":
                O = shape[it]
            if dims[d] == "H":
                H = shape[it]
            if dims[d] == "W":
                W = shape[it]
            if dims[d] == "C":
                C = shape[it]

            if dims[d] == "n":
                if digits != "":
                    if N != 0:
                        N *= int(digits)
                    else:
                        N = int(digits)
            
            if dims[d] == "h":
                if digits != "":
                    if H != 0:
                        H *= int(digits)
                    else:
                        H = int(digits)

            if dims[d] == "w":
                if digits != "":
                    if W != 0:
                        W *= int(digits)
                    else:
                        W = int(digits)

            if dims[d] == "c":
                if digits != "":
                    if C != 0:
                        C *= int(digits)
                    else:
                        C = int(digits)
            
            if dims[d] == "i":
                if digits != "":
                    if I != 0:
                        I *= int(digits)
                    else:
                        I = int(digits)
            
            if dims[d] == "o":
                if digits != "":
                    if O != 0:
                        O *= int(digits)
                    else:
                        O = int(digits)
            
            

            if dims[d].isdigit():
                digits += dims[d]
            else:
                it += 1
                digits = ""

        if N != 0:
            return N, C, H, W
        return O, I, H, W
                
    def from_relay_func(func, select="input"):
        ops = get_ops(func)
        if select == "input":
            op = ops[-1][1]
            if type(op.args[0]) == tvm.relay.expr.Tuple:
                print("concatenate")
                dtype = op.args[0].checked_type.fields[0].dtype
                shape = list(op.args[0].checked_type.fields[0].shape)
            else:
                shape = op.args[0].checked_type.concrete_shape
                dtype = op.args[0].checked_type.dtype

            if hasattr(func.attrs, 'data_layout'):
                layout = func.attrs.data_layout
            if hasattr(func.attrs, 'src_layout'):
                layout = func.attrs.src_layout
            else:
                layout = ""
            if "conv" in op.op.name :
                layout = op.attrs.data_layout
            if "pool" in op.op.name :
                layout = op.attrs.layout
            
            N, C, H, W = TensorConfig.extract_dims(shape, layout)
            I = None
            O = None
            

        if select == "output":
            op = ops[0][1]
            shape = op.checked_type.concrete_shape
            dtype = op.checked_type.dtype
            if hasattr(func.attrs, 'out_layout'):
                layout = func.attrs.out_layout
            else:
                layout = ""
            if hasattr(func.attrs, 'dst_layout'):
                layout = func.attrs.dst_layout

            N, C, H, W = TensorConfig.extract_dims(shape, layout)
            I = None
            O = None
            
        if select == "weight":
            op = ops[-1][1]
            if type(op.args[0]) == tvm.relay.expr.Tuple:
                print("concatenate")
                dtype = op.args[0].checked_type.fields[1].dtype
                shape = op.args[0].checked_type.fields[1].shape
            else:
                op = get_dominant_ops(func)[0][1]

            if "pool" in op.op.name :
                return TensorConfig(0, 0, 0, 0, "", "")
            
            if hasattr(func.attrs, 'kernel_layout'):
                layout = func.attrs.kernel_layout
            else:
                layout = ""
            
            shape = op.args[1].checked_type.concrete_shape
            dtype = op.args[1].checked_type.dtype

            O, I, H, W = TensorConfig.extract_dims(shape, layout)
            N = None
            C = None

        return TensorConfig(N, H, W, C, I, O, dtype, layout)
        
    def to_txt(self):
        txt = ""
        if hasattr(self, "N") and self.N is not None:
            txt += "N: "+str(self.N)+"\n"

        if hasattr(self, "C") and self.C is not None:
            txt += "C: "+str(self.C)+"\n"

        if hasattr(self, "O") and self.O is not None:
            txt += "O: "+str(self.O)+"\n"

        if hasattr(self, "I") and self.I is not None:
            txt += "I: "+str(self.I)+"\n"

        if hasattr(self, "H") and self.H is not None:
            txt += "H: "+str(self.H)+"\n"

        if hasattr(self, "W") and self.W is not None:
            txt += "W: "+str(self.W)+"\n"
        
        return txt

    def for_csv(self):
        values = []

        if hasattr(self, "N"):
            values.append(self.N)
        else:
            if hasattr(self, "O"):
                values.append(self.O)
            else:
                values.append(0)
        
        if hasattr(self, "C"):
            values.append(self.C)
        else:
            if hasattr(self, "I"):
                values.append(self.I)
            else:
                values.append(0)

        if hasattr(self, "H"):
            values.append(self.H)
        else:
            values.append(0)

        if hasattr(self, "W"):
            values.append(self.W)
        else:
            values.append(0)

        values.append(self.type)
        values.append(self.layout)

        return values
        
    #TODO: create from layer/ database entry/ ...

class CollectorInfo():

    head = [
        "Execution Time",
        "Workspace Size",
        "IO Size",
        "Power Consumption",
    ]

    def __init__(self, exec_time, workspace_size, io_size, power, power_inc_memory):
        if exec_time != None:
            self.execution_time = exec_time
        else:
            self.execution_time = float("inf")

        if workspace_size != None:
            self.workspace_size = workspace_size
        else:
            self.workspace_size = float("inf")
            
        if io_size != None:
            self.io_size = io_size
        else:
            self.io_size = float("inf")
        
        if power != None:
            self.power_consumption = power
        else:
            self.power_consumption = float("inf")
        if power_inc_memory != None:
            self.power_including_memory = power_inc_memory
        else:
            self.power_including_memory = False
            
        return

    #TODO: create from layer/ database entry/ ...
    def get_scheme(self):
        values = dict()
        if self != None:
            values["Execution Time"] = self.execution_time
            values["Workspace Size"] = self.workspace_size
            values["IO size"] = self.io_size
            values["Power Consumption"] = self.power_consumption
        else:
            values["Execution Time"] = float("inf")
            values["Workspace Size"] = float("inf")
            values["IO size"] = float("inf")
            values["Power Consumption"] = float("inf")
        return values

    def to_txt(self):
        txt = ""
        txt += "execution_time: "+ str(self.execution_time) + "\n"
        txt += "workspace_size: "+ str(self.workspace_size) + "\n"
        txt += "io_size: "+ str(self.io_size) + "\n"
        txt += "power_consumption: "+ str(self.power_consumption) + "\n"
        txt += "power_includes_local_mem: "+ str(self.power_including_memory) + "\n"

        return txt
    
    def for_csv(self):
        values = [self.execution_time, self.workspace_size, self.io_size, self.power_consumption, self.power_including_memory]
        return values

class DescriptionVector():

    known_values = dict()

    def __init__(self, debug_data, op_desc, op_config, input_config, weight_config, output_config, collector_info = None):
       
        if type(debug_data) == DebugInfo:
            self.debug_data = debug_data
        else:
            self.debug_data = None

        if type(op_desc) == OperationDescription:
            self.op_desc = op_desc
        else:
            self.op_desc = None
        
        if type(op_config) == OperatorConfig:
            self.op_config = op_config
        else:
            self.op_config = None

        if type(input_config) == TensorConfig:
            self.input_config = input_config
        else:
            self.input_config = None

        if type(weight_config) == TensorConfig:
            self.weight_config = weight_config
        else:
            self.weight_config = None

        if type(output_config) == TensorConfig:
            self.output_config = output_config
        else:
            self.output_config = None

        if type(collector_info) == CollectorInfo:
            self.collector_infos = [collector_info]
        else:
            self.collector_infos = list()

        if type(collector_info) == list:
            self.collector_infos = collector_info
        return

    def to_txt(self):
        txt = ""
        if self.debug_data != None:
            txt += "Debug  Information:\n"
            txt += self.debug_data.to_txt() + "\n"
        
        if self.op_desc != None:
            txt += "Operation Description:\n"
            txt += self.op_desc.to_txt() + "\n"

        if self.op_config != None:
            txt += "Operator Config:\n"
            txt += self.op_config.to_txt() + "\n"

        if self.input_config != None:
            txt += "Input Tensor Config:\n"
            txt += self.input_config.to_txt() + "\n"

        if self.output_config != None:
            txt += "Output Tensor Config:\n"
            txt += self.output_config.to_txt() + "\n"
        
        if self.weight_config != None:
            txt += "Weight Tensor Config:\n"
            txt += self.weight_config.to_txt() + "\n"

        for i, data in enumerate(self.collector_infos):
            txt += "Collected Run #"+str(i)+":\n"
            txt += data.to_txt()+"\n"

        return txt

    def csv_head():
        headers = []
        headers += DebugInfo.head
        headers += OperationDescription.head
        headers += OperatorConfig.head
        #headers += TensorConfig.head
        #headers += TensorConfig.head
        #headers += TensorConfig.head
        headers.append("input N")
        headers.append("input C")
        headers.append("input H")
        headers.append("input W")
        headers.append("input type")
        headers.append("input layout")

        headers.append("weight O")
        headers.append("weight I")
        headers.append("weight H")
        headers.append("weight W")
        headers.append("weight type")
        headers.append("weight layout")

        headers.append("output N")
        headers.append("output C")
        headers.append("output H")
        headers.append("output W")
        headers.append("output type")
        headers.append("output layout")
        headers += CollectorInfo.head

        return headers

    def get_scheme(self):
        
        values = dict()
        values.update(DebugInfo.get_scheme(self.debug_data))
        values.update(OperationDescription.get_scheme(self.op_desc))
        values.update(OperatorConfig.get_scheme(self.op_config))

        values.update(TensorConfig.get_scheme(self.input_config, "input"))
        values.update(TensorConfig.get_scheme(self.weight_config, "weight"))
        values.update(TensorConfig.get_scheme(self.output_config, "output"))
        
        if len(self.collector_infos)  == 0:
            values["Execution Times"] = []
            values["Workspace Sizes"] = []
            values["IO Sizes"] = []
            values["Power Consumptions"] = []
        else:
            times = []
            workspaces = []
            ios = []
            powers = []
            inc_mems = []
            for infos in self.collector_infos:
                times.append(infos.execution_time)
                workspaces.append(infos.workspace_size)
                ios.append(infos.io_size)
                powers.append(infos.power_consumption)
                inc_mems.append(infos.power_including_memory)
         
            values["Execution Time"] = times
            values["Workspace Size"] = workspaces
            values["IO Size"] = ios
            values["Power Consumption"] = powers

        return values
    
    def for_csv(self):
        values = list()
        if self.debug_data != None:
            values += self.debug_data.for_csv()
        else:
            values += ["", ""]

        if self.op_desc != None:
            values += self.op_desc.for_csv()
        else:
            values += ["", "", 0, 0, 0]

        if self.op_config != None:
            values += self.op_config.for_csv()
        else:
            values += [[0, 0], [0,0,0,0], [0,0], [0,0], 0, 0, [0,0]]

        if self.input_config != None:
            values += self.input_config.for_csv()
        else:
            values += [0, 0, 0, 0, "", ""]

        if self.weight_config != None:
            values += self.weight_config.for_csv()
        else:
            values += [0, 0, 0, 0, "", ""]
        
        if self.output_config != None:
            values += self.output_config.for_csv()
        else:
            values += [0, 0, 0, 0, "", ""]
        
        if len(self.collector_infos)  == 0:
            values += [0, 0, 0, 0, "False"]
        else:
            times = []
            workspaces = []
            ios = []
            powers = []
            inc_mems = []
            for infos in self.collector_infos:
                times.append(infos.execution_time)
                workspaces.append(infos.workspace_size)
                ios.append(infos.io_size)
                powers.append(infos.power_consumption)
                inc_mems.append(infos.power_including_memory)
            values += [times, workspaces, ios, powers, inc_mems]

        return values
            

