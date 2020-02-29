import onnx


_PREFIX = 'OMMLE'


def get_new_name(basename, model):
    connection_names = [i.name for i in model.graph.input] + [o.name for o in model.graph.output] + [o for n in model.graph.node for o in n.output]

    i = 0
    while f'{_PREFIX}.{basename}_{i}' in connection_names:
        i += 1

    return f'{_PREFIX}.{basename}_{i}'


def add_middle_outputs(original_model, cast_type=None, exclude_op_types=None, exclude_output_names=None):
    if not exclude_op_types:
        exclude_op_types = []
    if not exclude_output_names:
        exclude_output_names = []

    modified_model = onnx.helper.make_model(original_model.graph)

    onnx_output_names = [o.name for o in modified_model.graph.output]
    for node in list(modified_model.graph.node):
        if node.op_type in exclude_op_types:
            continue
        for output_name in node.output:
            if output_name in exclude_output_names:
                continue
            if output_name in onnx_output_names:
                continue
            if cast_type:
                new_output_name = get_new_name(output_name + '.cast', modified_model)
                cast_node = onnx.helper.make_node('Cast', inputs=[output_name], outputs=[new_output_name], to=cast_type)
                modified_model.graph.node.append(cast_node)
                new_value_info = onnx.helper.make_tensor_value_info(new_output_name, cast_type, None)
            else:
                new_output_name = get_new_name(output_name + '.middle', modified_model)
                identity_node = onnx.helper.make_node('Identity', inputs=[output_name], outputs=[new_output_name])
                modified_model.graph.node.append(identity_node)
                new_value_info = onnx.helper.make_tensor_value_info(new_output_name, onnx.TensorProto.FLOAT, None)
            modified_model.graph.output.append(new_value_info)

    return modified_model

