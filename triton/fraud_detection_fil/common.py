import os

REPO_PATH = os.path.abspath('../')

def get_model_dir(model_name):
    model_dir = os.path.join(REPO_PATH, model_name)
    return model_dir

small_model_dir = get_model_dir('small_model')
small_model_cpu_dir = get_model_dir('small_model-cpu')
large_model_dir = get_model_dir('large_model')
large_model_cpu_dir = get_model_dir('large_model-cpu')

X_test_file_name='X_test.pickle'
y_test_file_name='y_test.pickle'
small_model_file_name='small_model.pickle'
large_model_file_name='large_model.pickle'
