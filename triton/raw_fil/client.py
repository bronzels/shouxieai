import time
import tritonclient.grpc as triton_grpc
from tritonclient import utils as triton_utils

import pickle

HOST = 'localhost'
PORT = 8001
TIMEOUT = 60

client = triton_grpc.InferenceServerClient(url=f'{HOST}:{PORT}')

# Wait for server to come online
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

X_test_file_name='X_test.pickle'
with open(X_test_file_name, 'rb') as X_test_file:
    X_test = pickle.load(X_test_file)

np_data = convert_to_numpy(X_test).astype('float32')

def triton_predict(model_name, arr):
    triton_input = triton_grpc.InferInput('input__0', arr.shape, 'FP32')
    triton_input.set_data_from_numpy(arr)
    triton_output = triton_grpc.InferRequestedOutput('output__0')
    response = client.infer(model_name, model_version='1', inputs=[triton_input], outputs=[triton_output])
    return response.as_numpy('output__0')

triton_result = triton_predict('small_model', np_data[0:5])
from common import small_model_file_name
with open(small_model_file_name, 'rb') as small_model_file:
    small_model = pickle.load(small_model_file)
local_result = small_model.predict_proba(X_test[0:5])
print("Result computed on Triton: ")
print(triton_result)
print("\nResult computed locally: ")
print(local_result)
cp.testing.assert_allclose(triton_result, local_result, rtol=1e-6, atol=1e-6)

