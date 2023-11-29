import cupy as cp
import os
import pickle

MAX_MEMORY_BYTES = 60000000
from common import X_test_file_name
with open(X_test_file_name, 'rb') as X_test_file:
    X_test = pickle.load(X_test_file)
features = X_test.shape[1]
from common import y_test_file_name
with open(y_test_file_name, 'rb') as y_test_file:
    y_test = pickle.load(y_test_file)
num_classes = cp.unique(y_test).size
bytes_per_sample = (features + num_classes) * 4
max_batch_size = MAX_MEMORY_BYTES // bytes_per_sample

def generate_config(model_dir, deployment_type='gpu', storage_type='AUTO'):
    if deployment_type.lower() == 'cpu':
        instance_kind = 'KIND_CPU'
    else:
        instance_kind = 'KIND_GPU'

    config_text = f"""backend: "fil"
max_batch_size: {max_batch_size}
input [                                 
 {{  
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {features} ]                    
  }} 
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
  }}
]
instance_group [{{ kind: {instance_kind} }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
  }},
  {{
    key: "predict_proba"
    value: {{ string_value: "true" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "true" }}
  }},
  {{
    key: "threshold"
    value: {{ string_value: "0.5" }}
  }},
  {{
    key: "storage_type"
    value: {{ string_value: "{storage_type}" }}
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: 100
}}
"""
    config_path = os.path.join(model_dir, 'config.pbtxt')
    with open(config_path, 'w') as file_:
        file_.write(config_text)

    return config_path

from common import small_model_dir, small_model_cpu_dir, large_model_dir, large_model_cpu_dir

generate_config(small_model_dir, deployment_type='gpu')
generate_config(small_model_cpu_dir, deployment_type='cpu')
generate_config(large_model_dir, deployment_type='gpu')
generate_config(large_model_cpu_dir, deployment_type='cpu')




