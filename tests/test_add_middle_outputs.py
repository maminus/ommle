import onnx
import onnx.numpy_helper
import numpy as np
from onnxruntime import InferenceSession
from ommle import add_middle_outputs


def make_simple_model():
    # input_0 -> Conv -> Relu -> Concat -> output_0
    #          \--------------/
    inputs = [
        onnx.helper.make_tensor_value_info('input_0', onnx.TensorProto.FLOAT, ['B', 3, 2, 2]),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info('output_0', onnx.TensorProto.FLOAT, ['B', 7, 2, 2]),
    ]
    nodes = [
        onnx.helper.make_node('Conv', inputs=['input_0', 'conv_1.weight'], outputs=['conv_1']),
        onnx.helper.make_node('Relu', inputs=['conv_1'], outputs=['relu_1']),
        onnx.helper.make_node('Concat', inputs=['relu_1', 'input_0'], outputs=['output_0'], axis=1),
    ]
    weight_shape = [4, 3, 1, 1]
    inits = [
        onnx.numpy_helper.from_array(np.arange(np.prod(weight_shape), dtype=np.float32).reshape(weight_shape), 'conv_1.weight') 
    ]
    graph = onnx.helper.make_graph(nodes, 'simple_model', inputs, outputs, inits)
    model = onnx.helper.make_model(graph)

    return \
        model, \
        onnx.helper.make_model(onnx.helper.make_graph(nodes[:1], 'conv', inputs, [onnx.helper.make_tensor_value_info('conv_1', onnx.TensorProto.FLOAT, ['B', 4, 2, 2])], inits)), \
        onnx.helper.make_model(onnx.helper.make_graph(nodes[:2], 'relu', inputs, [onnx.helper.make_tensor_value_info('relu_1', onnx.TensorProto.FLOAT, ['B', 4, 2, 2])], inits))


def test_default():
    original_model, conv, relu = make_simple_model()
    modified_model = add_middle_outputs(original_model)

    print(onnx.helper.printable_graph(modified_model.graph))

    input_shape = [5, 3, 2, 2]
    data = np.ones(input_shape, dtype=np.float32)

    sess = InferenceSession(modified_model.SerializeToString())
    outputs = sess.run([o.name for o in sess.get_outputs()], {i.name: data for i in sess.get_inputs()})

    sess = InferenceSession(conv.SerializeToString())
    conv_output = sess.run([o.name for o in sess.get_outputs()], {i.name: data for i in sess.get_inputs()})

    sess = InferenceSession(relu.SerializeToString())
    relu_output = sess.run([o.name for o in sess.get_outputs()], {i.name: data for i in sess.get_inputs()})

    assert np.all(outputs[1] == conv_output[0])
    assert np.all(outputs[2] == relu_output[0])


def test_cast_type():
    original_model, _, _ = make_simple_model()
    modified_model = add_middle_outputs(original_model, cast_type=onnx.TensorProto.FLOAT16)

    input_shape = [5, 3, 2, 2]
    data = np.ones(input_shape, dtype=np.float32)

    sess = InferenceSession(modified_model.SerializeToString())
    outputs = sess.run([o.name for o in sess.get_outputs()], {i.name: data for i in sess.get_inputs()})

    assert outputs[0].dtype == np.float32
    assert outputs[1].dtype == np.float16
    assert outputs[2].dtype == np.float16


def test_exclude_op_types():
    original_model, _, _ = make_simple_model()
    modified_model = add_middle_outputs(original_model, exclude_op_types=['Relu'])

    output_names = [o.name for o in modified_model.graph.output]
    assert 'relu' not in output_names


def test_exclude_output_names():
    original_model, _, _ = make_simple_model()
    modified_model = add_middle_outputs(original_model, exclude_output_names=['relu_1'])

    output_names = [o.name for o in modified_model.graph.output]
    assert 'OMMLE.relu_1.middle_0' not in output_names

