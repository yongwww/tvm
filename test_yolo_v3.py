import numpy as np
import tensorflow as tf
import tvm
import time
from tvm import relay
from tvm.relay.frontend.tensorflow_parser import TFParser

def run_vm(model_path):

    input_name = "image_tensor"
    input_shape = (1, 512, 512, 3)
    graph_def = TFParser(model_path).parse()
    mod, params = relay.frontend.from_tensorflow(graph_def, shape={input_name: input_shape}, layout="NCHW")

    print("Relay IR: {}".format(mod))
    import time
    start_time = time.time()
    print("staring to build")
    with relay.build_config(opt_level=3):#, disabled_pass=["AlterOpLayout"]):
        exe = relay.vm.compile(mod, "llvm", params=params)
    vm = relay.vm.VirtualMachine(exe)
    vm_create_time = time.time()
    vm.init(tvm.cpu())
    data = np.random.randint(250, size=input_shape).astype("uint8")
    print("starting to do inference")
    vm_ret = vm.run(data)
    print("vm result: ", vm_ret.asnumpy())
    print("############# FINISHED ############")

def benchmark(model_path, inputs={}, outputs=[]):
    def run_tf_graph(sess, input_data, input_node, output_node):
        """ Generic function to execute tensorflow """
        def convert_to_list(x):
            if not isinstance(x, list):
                x = [x]
            return x
        input_data = convert_to_list(input_data)
        input_node = convert_to_list(input_node)
        output_node = convert_to_list(output_node)

        tensor = [sess.graph.get_tensor_by_name(
            output_name) for output_name in output_node]

        input_dict = {e: input_data[i] for i, e in enumerate(input_node)}

        for i in range(10):
            output_data = sess.run(tensor, input_dict)
        tf_start_time = time.time()
        for i in range(50):
            output_data = sess.run(tensor, input_dict)
        tf_end_time = time.time()
        tf_exec_time = (tf_end_time - tf_start_time) * 1000 / 50
        print("TensorFlow execution time: ", tf_exec_time)

        return output_data

    def _load_frozen_graph(frozen_graph_file):
        """
        Load Frozen Graph

        Parameters
        ----------
        frozen_graph_file : str
            Full path to frozen graph (.pb file)
        device : str
            device type and id, (e.g. /cpu:0)

        Returns
        -------
        out : Graph :py:class:`tf.Graph`
        """
        with tf.gfile.GFile(frozen_graph_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=PREFIX)
        return graph

    def _get_input_and_output_names(graph):
        """
        Get input and output tensor names

        Parameters
        ----------
        graph : tf.Graph
            Tensorflow graph

        Returns
        -------
        input_tensor_names : List of input tensor names
        output_tensor_names  : List of output tensor names
        """
        input_tensor_names = []
        output_tensor_names = set()
        op_prefix = PREFIX + "/"
        for op in graph.get_operations():
            if not op.name.startswith(op_prefix):
                continue
            if op.type == 'Placeholder' and op.inputs.__len__() == 0 and op.outputs.__len__() == 1:
                input_tensor_names.append(op.outputs[0].name)
            if op.type not in UNLIKELY_OUTPUT_TYPES and op.outputs.__len__() == 1:
                output_tensor_names.add(op.outputs[0].name)

        for op in graph.get_operations():
            for in_t in op.inputs:
                if in_t.name in output_tensor_names:
                    output_tensor_names.remove(in_t.name)
            for cont_op in op.control_inputs:
                for out_t in cont_op.outputs:
                    if out_t.name in output_tensor_names:
                        output_tensor_names.remove(out_t.name)
        # Sort list of output tensor names in order to get consistent output in run()
        output_tensor_names = list(output_tensor_names)
        output_tensor_names.sort()
        return input_tensor_names, output_tensor_names
    if len(inputs) == 0: # SSD, Fast-RCNN, Faster-RCNN, Mask-RCNN
        inputs = {'image_tensor': (1, 512, 512, 3)}
        input_name = "image_tensor"
        input_shape = (1, 512, 512, 3)
    else: # yolo
        input_name = "input/input_data"
        input_shape = (1, 416, 416, 3)

    data = np.random.randint(250, size=input_shape).astype("float32")


    print("Starting to parse the graph to relay ir")
    graph_def = TFParser(model_path).parse()
    outs = [output[7:] if output.startswith('import/') else output for output in outputs ]
    mod, params = relay.frontend.from_tensorflow(graph_def, shape={input_name: input_shape},
                                                 outputs=outs,
                                                 layout="NCHW")

    print("############# Starting to run with TVM Relay VM ############")
    # print("Relay IR: {}".format(mod))
    print("starting to build")
    tgt = "llvm -mcpu=skylake-avx512"
    with relay.build_config(opt_level=3):
        exe = relay.vm.compile(mod, tgt, params=params)
    print("build done")
    vm = relay.vm.VirtualMachine(exe)
    vm.init(tvm.cpu())

    print("starting to do inference and measurement")
    # warm up
    for i in range(5):
        vm_ret = vm.run(data)
    vm_start_time = time.time()
    # measure execution time
    for i in range(50):
        vm_ret = vm.run(data)
    vm_end_time = time.time()
    vm_exec_time = (vm_end_time - vm_start_time) * 1000 / 50
    print("vm execution time: ", vm_exec_time)
    print("vm result: ", vm_ret.asnumpy())

    print("############# Finish inference with TVM Relay VM ############")


    print("############# Starting to run with TVM Graph Runtime ############")

    # print("Relay IR: {}".format(mod))
    print("starting to build")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", None, params)

    ctx = tvm.context("llvm", 0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(data))

    m.set_input(**params)


    # warm up
    for i in range(5):
        # execute
        m.run()
    graph_start_time = time.time()
    # measure execution time
    for i in range(50):
        # execute
        m.run()
    graph_end_time = time.time()
    graph_exec_time = (graph_end_time - graph_start_time) * 1000 / 50
    print("graph runtime execution time: ", graph_exec_time)
    tvm_output_list = m.get_output(0).asnumpy()
    print("ret of graph runtime: ", tvm_output_list)

    print("############# Finish inference with TVM Graph Runtime ############")


    print("############# Starting to run with TensorFlow ############")
    # A prefix that will be prepended to the names in graph_def
    PREFIX = "import"
    UNLIKELY_OUTPUT_TYPES = {"Const", "Assign", "NoOp", "Placeholder"}
    _graph = _load_frozen_graph(model_path)
    input_tensor_names, output_tensor_names = _get_input_and_output_names(_graph)
    print("input_tensor_names: {}\n, output_tensor_names: {}"
          .format(input_tensor_names, output_tensor_names))
    with tf.compat.v1.Session(graph=_graph) as sess:
        tf_output = run_tf_graph(
            sess, data, list(inputs.keys())[0], outputs[0])
        print("tf_output: {}".format(tf_output))

    print("############# Finish inference with TensorFlow ############")

    tvm.testing.assert_allclose(vm_ret.asnumpy(), tf_output[0],  rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    # yolo-v3: https://drive.google.com/file/d/1UfQyGw4Y8HOjTlOx0Mq1V1BpREPlBgAc/view?usp=sharing
    yolo_inputs = {'import/input/input_data:0': (1, 416, 416, 3)}
    # yolo_outs = ['import/pred_lbbox/concat_2:0', 'import/pred_mbbox/concat_2:0', 'import/pred_sbbox/concat_2:0']
    yolo_outs = ['import/pred_sbbox/concat_2:0']
    benchmark("/Users/yongwu/Desktop/TVM/models/OD/yolo/yolov3_coco.pb", yolo_inputs, yolo_outs)
