import tensorflow as tf
import tensorflow.python.keras.backend
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def h5_to_pb(model, pb_model_name):
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2((full_model))
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./",
                      name=pb_model_name,
                      as_text=False)

h5_model_path = "./weights.h5"
pb_model_name = "weights.pb"

net_model = load_model(h5_model_path)
h5_to_pb(net_model, pb_model_name)

"""
pip install tf2onnx
python -m tf2onnx.convert --graphdef weights.pb --output weights.onnx --inputs Input:0 --outputs Identity:0
"""
