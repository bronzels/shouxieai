import time
import tritonclient.grpc as triton_grpc
from tritonclient import utils as triton_utils
import cupy as cp
import pickle

HOST = 'localhost'
PORT = 8001
TIMEOUT = 60

client = triton_grpc.InferenceServerClient(url=f'{HOST}:{PORT}')

server_start = time.time()
while True:
    try:
        if client.is_server_ready() or time.time() - server_start > TIMEOUT:
            break
    except triton_utils.InferenceServerException:
        pass
    time.sleep(1)

import pandas as pd
def convert_to_numpy(df):
    df = df.copy()
    cat_cols = df.select_dtypes('category').columns
    for col in cat_cols:
        df[col] = df[col].cat.codes
    for col in df.columns:
        df[col] =  pd.to_numeric(df[col], downcast='float')
    return df.values

from common import X_test_file_name
with open(X_test_file_name, 'rb') as X_test_file:
    X_test = pickle.load(X_test_file)

np_data = convert_to_numpy(X_test).astype('float32')

def triton_predict(model_name, arr):
    triton_input = triton_grpc.InferInput('input__0', arr.shape, 'FP32')
    triton_input.set_data_from_numpy(arr)
    triton_output = triton_grpc.InferRequestedOutput('output__0')
    response = client.infer(model_name, model_version='1', inputs=[triton_input], outputs=[triton_output])
    return response.as_numpy('output__0')

for model_size in ["small_model", "large_model"]:
    with open('{}.pickle'.format(model_size), 'rb') as model_file:
        model = pickle.load(model_file)
    print('\n\n\n\nmodel_size:', model_size)
    local_result = model.predict_proba(X_test[0:5])
    print("\nResult computed locally: ")
    print(local_result)
    for plat in ["cpu", "gpu"]:
        model_name = '{}-{}'.format(model_size, plat)
        print('model_name:', model_name)
        a = time.time()
        triton_result = triton_predict('small_model', np_data[0:5])
        b = time.time()
        print("infer time = %f" % (b - a))
        print("Result computed on Triton: ")
        print(triton_result)
        try:
            cp.testing.assert_allclose(triton_result, local_result, rtol=1e-6, atol=1e-6)
        except AssertionError as arg:
            print('%s, %s' % (arg.__class__.__name__, arg))
        print("\n\n\n")
"""
curl -X POST http://localhost:8000/v2/repository/models/fraud_detection_fil/small_model/load
curl -X POST http://localhost:8000/v2/repository/models/fraud_detection_fil/small_model-cpu/load
curl -X POST http://localhost:8000/v2/repository/models/fraud_detection_fil/large_model/load
curl -X POST http://localhost:8000/v2/repository/models/fraud_detection_fil/large_model-cpu/load

python client.py


model_size: small_model
/root/miniconda3/envs/triton_example/lib/python3.8/site-packages/xgboost/data.py:290: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead.  To get a de-fragmented frame, use `newframe = frame.copy()`
  transformed[data.columns[i]] = data[data.columns[i]]

Result computed locally: 
[[0.98954797 0.01045203]
 [0.99720675 0.00279326]
 [0.9911508  0.00884922]
 [0.97535217 0.02464783]
 [0.99197257 0.00802745]]
model_name: small_model-cpu
infer time = 0.003060
Result computed on Triton: 
[[0.99291897 0.00708104]
 [0.9974075  0.00259252]
 [0.992248   0.007752  ]
 [0.94361895 0.05638104]
 [0.99307674 0.00692324]]
AssertionError, 
Not equal to tolerance rtol=1e-06, atol=1e-06

Mismatched elements: 10 / 10 (100%)
Max absolute difference: 0.03173321
Max relative difference: 1.2874647
 x: array([[0.992919, 0.007081],
       [0.997407, 0.002593],
       [0.992248, 0.007752],...
 y: array([[0.989548, 0.010452],
       [0.997207, 0.002793],
       [0.991151, 0.008849],...




model_name: small_model-gpu
infer time = 0.000603
Result computed on Triton: 
[[0.99291897 0.00708104]
 [0.9974075  0.00259252]
 [0.992248   0.007752  ]
 [0.94361895 0.05638104]
 [0.99307674 0.00692324]]
AssertionError, 
Not equal to tolerance rtol=1e-06, atol=1e-06

Mismatched elements: 10 / 10 (100%)
Max absolute difference: 0.03173321
Max relative difference: 1.2874647
 x: array([[0.992919, 0.007081],
       [0.997407, 0.002593],
       [0.992248, 0.007752],...
 y: array([[0.989548, 0.010452],
       [0.997207, 0.002793],
       [0.991151, 0.008849],...








model_size: large_model

Result computed locally: 
[[9.9999619e-01 3.7907150e-06]
 [9.9999982e-01 1.8070210e-07]
 [9.9999124e-01 8.7775970e-06]
 [9.9997872e-01 2.1281312e-05]
 [9.9999988e-01 1.0316177e-07]]
model_name: large_model-cpu
infer time = 0.001011
Result computed on Triton: 
[[0.99291897 0.00708104]
 [0.9974075  0.00259252]
 [0.992248   0.007752  ]
 [0.94361895 0.05638104]
 [0.99307674 0.00692324]]
AssertionError, 
Not equal to tolerance rtol=1e-06, atol=1e-06

Mismatched elements: 10 / 10 (100%)
Max absolute difference: 0.05635977
Max relative difference: 67109.51
 x: array([[0.992919, 0.007081],
       [0.997407, 0.002593],
       [0.992248, 0.007752],...
 y: array([[9.999962e-01, 3.790715e-06],
       [9.999998e-01, 1.807021e-07],
       [9.999912e-01, 8.777597e-06],...




model_name: large_model-gpu
infer time = 0.000597
Result computed on Triton: 
[[0.99291897 0.00708104]
 [0.9974075  0.00259252]
 [0.992248   0.007752  ]
 [0.94361895 0.05638104]
 [0.99307674 0.00692324]]
AssertionError, 
Not equal to tolerance rtol=1e-06, atol=1e-06

Mismatched elements: 10 / 10 (100%)
Max absolute difference: 0.05635977
Max relative difference: 67109.51
 x: array([[0.992919, 0.007081],
       [0.997407, 0.002593],
       [0.992248, 0.007752],...
 y: array([[9.999962e-01, 3.790715e-06],
       [9.999998e-01, 1.807021e-07],
       [9.999912e-01, 8.777597e-06],...

"""
