# Maximum size in bytes for input and output arrays. If you are
# using Triton 21.11 or higher, all memory allocations will make
# use of Triton's memory pool, which has a default size of
# 67_108_864 bytes. This can be increased using the
# `--cuda-memory-pool-byte-size` option when the server is
# started, but this notebook should work fine with default
# settings.
MAX_MEMORY_BYTES = 60_000_000

features = X_test.shape[1]
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
}}"""
    config_path = os.path.join(model_dir, 'config.pbtxt')
    with open(config_path, 'w') as file_:
        file_.write(config_text)

    return config_path

generate_config(small_model_dir, deployment_type='gpu')
generate_config(small_model_cpu_dir, deployment_type='cpu')
generate_config(large_model_dir, deployment_type='gpu')
generate_config(large_model_cpu_dir, deployment_type='cpu')

