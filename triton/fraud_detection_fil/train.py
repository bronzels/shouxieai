"""
wget -O /opt/miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    yes
    <enter>
    yes

conda config --remove-key channels

conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/

conda update --all -y
conda update -n base conda

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#numpy                     1.24.4
#ImportError: cannot import name 'MachAr' from 'numpy'
#修改conda.yaml里numpy版本
conda env create -f conda.yaml
conda activate triton_example
pip install ptxcompiler-cu11 cubinlinker-cu11 --extra-index-url=https://pypi.nvidia.com

#下载kaggle.json
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c ieee-fraud-detection
unzip -u ieee-fraud-detection.zip
ls *.csv

"""
from numba import config
config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = 1
USE_CATEGORICAL = True
train_csv = 'train_transaction.csv'
import cudf
import cupy as cp
from cuml.preprocessing import SimpleImputer
if not USE_CATEGORICAL:
    from cuml.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pickle

SEED = 0

data = cudf.read_csv(train_csv)

nan_columns = data.columns[data.isna().any().to_pandas()]
float_nan_subset = data[nan_columns].select_dtypes(include='float64')

imputer = SimpleImputer(missing_values=cp.nan, strategy='mean')
data[float_nan_subset.columns] = imputer.fit_transform(float_nan_subset)

obj_nan_subset = data[nan_columns].select_dtypes(include='object')
data[obj_nan_subset.columns] = obj_nan_subset.fillna('UNKNOWN')

cat_columns = data.select_dtypes(include='object')
if USE_CATEGORICAL:
    data[cat_columns.columns] = cat_columns.astype('category')
else:
    for col in cat_columns.columns:
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('isFraud', axis=1)
y = data.isFraud.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X.to_pandas(), y.to_pandas(), test_size=0.3, stratify=y.to_pandas(), random_state=SEED
)
from common import y_test_file_name
with open(y_test_file_name, 'wb') as y_test_file:
    pickle.dump(y_test, y_test_file)
X_train = X_train.copy()
X_test = X_test.copy()
from common import X_test_file_name
with open(X_test_file_name, 'wb') as X_test_file:
    pickle.dump(X_test, X_test_file)

import xgboost as xgb

def train_model(num_trees, max_depth, model_name):
    model = xgb.XGBClassifier(
        tree_method='gpu_hist',
        enable_categorical=USE_CATEGORICAL,
        use_label_encoder=False,
        predictor='gpu_predictor',
        eval_metric='aucpr',
        objective='binary:logistic',
        max_depth=max_depth,
        n_estimators=num_trees
    )
    print("------start of {} training".format(model_name))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)]
    )
    print("------end of {} training".format(model_name))
    return model


from common import small_model_file_name, large_model_file_name

small_model = train_model(500, 3, "small_model")
with open(small_model_file_name, 'wb') as small_model_file:
    pickle.dump(small_model, small_model_file)

large_model = train_model(5000, 12, "large_model")
with open(large_model_file_name, 'wb') as large_model_file:
    pickle.dump(large_model, large_model_file)

import gc
del data
del nan_columns
del float_nan_subset
del imputer
del obj_nan_subset
del cat_columns
del X
del y
gc.collect()

import os
from common import REPO_PATH, small_model_dir, small_model_cpu_dir, large_model_dir, large_model_cpu_dir

os.makedirs(REPO_PATH, exist_ok=True)

def serialize_model(model, model_dir):
    version_dir = os.path.join(model_dir, '1')
    os.makedirs(version_dir, exist_ok=True)

    model_file = os.path.join(version_dir, 'xgboost.json')
    model.save_model(model_file)

serialize_model(small_model, small_model_dir)
serialize_model(small_model, small_model_cpu_dir)
serialize_model(large_model, large_model_dir)
serialize_model(large_model, large_model_cpu_dir)

"""
python train.py

"""
