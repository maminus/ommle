# ommle
ONNX Model's Middle Layer Extractor

This is a Python library which extracts the middle-layer outputs of an ONNX model.

## Installation
```
$ pip install onnx
$ pip install git+https://github.com/maminus/ommle.git
```

## Usage
```python
import onnx
from onnxruntime import InferenceSession    # for inference
from ommle import add_middle_outputs

# load an onnx model file
original_model = onnx.load('something-model.onnx')

# add middle-layer outputs to loaded model
modified_model = add_middle_outputs(original_model)

# now we have an additional ValueInfoProto to modified_model.graph.output[...]

sess = InferenceSession(modified_model.SerializeToString())

# prepare input data
input_name = sess.get_inputs()[0]
input_data = ...
input_dict = {input_name: input_data}

# do inference
output_list = sess.run([o.name for o in sess.get_outputs()], input_dict)

# output is a list. first is normal ONNX output (numpy.ndarray).
output_list[0]

# following elements are middle-layer outputs.
output_list[...]
```

