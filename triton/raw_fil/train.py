USE_CATEGORICAL = True
train_csv = 'train_transaction.csv'

import cudf
import cupy as cp
from cuml.preprocessing import SimpleImputer

if not USE_CATEGORICAL:
    from cuml.preprocessing import LabelEncoder
# Due to an upstream bug, cuML's train_test_split function is
# currently non-deterministic. We will therefore use sklearn's
# train_test_split in this example to obtain more consistent
# results.
from sklearn.model_selection import train_test_split

import pickle

SEED = 0
# Load data from CSV files into cuDF DataFrames
data = cudf.read_csv(train_csv)
# Replace NaNs in data
nan_columns = data.columns[data.isna().any().to_pandas()]
float_nan_subset = data[nan_columns].select_dtypes(include='float64')

imputer = SimpleImputer(missing_values=cp.nan, strategy='mean')
data[float_nan_subset.columns] = imputer.fit_transform(float_nan_subset)

obj_nan_subset = data[nan_columns].select_dtypes(include='object')
data[obj_nan_subset.columns] = obj_nan_subset.fillna('UNKNOWN')
# Convert string columns to categorical or perform label encoding
cat_columns = data.select_dtypes(include='object')
if USE_CATEGORICAL:
    data[cat_columns.columns] = cat_columns.astype('category')
else:
    for col in cat_columns.columns:
        data[col] = LabelEncoder().fit_transform(data[col])
# Split data into training and testing sets
X = data.drop('isFraud', axis=1)
y = data.isFraud.astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X.to_pandas(), y.to_pandas(), test_size=0.3, stratify=y.to_pandas(), random_state=SEED
)
# Copy data to avoid slowdowns due to fragmentation
X_train = X_train.copy()
X_test = X_test.copy()
from common import X_test_file_name
with open(X_test_file_name, 'wb') as X_test_file:
    pickle.dump(X_test, X_test_file)

import xgboost as xgb


# Define model training function
def train_model(num_trees, max_depth):
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
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)]
    )
    return model


# Train a small model with just 500 trees and a maximum depth of 3
small_model = train_model(500, 3)
small_model_file_name='small_model.pickle'
with open(small_model_file_name, 'wb') as small_model_file:
    pickle.dump(small_model, small_model_file)
# Train a large model with 5000 trees and a maximum depth of 12
large_model = train_model(5000, 12)
# Free up some room on the GPU by explicitly deleting dataframes
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

# Create the model repository directory. The name of this directory is arbitrary.
REPO_PATH = os.path.abspath('model_repository')
os.makedirs(REPO_PATH, exist_ok=True)


def serialize_model(model, model_name):
    # The name of the model directory determines the name of the model as reported
    # by Triton
    model_dir = os.path.join(REPO_PATH, model_name)
    # We can store multiple versions of the model in the same directory. In our
    # case, we have just one version, so we will add a single directory, named '1'.
    version_dir = os.path.join(model_dir, '1')
    os.makedirs(version_dir, exist_ok=True)

    # The default filename for XGBoost models saved in json format is 'xgboost.json'.
    # It is recommended that you use this filename to avoid having to specify a
    # name in the configuration file.
    model_file = os.path.join(version_dir, 'xgboost.json')
    model.save_model(model_file)

    return model_dir

small_model_dir = serialize_model(small_model, 'small_model')
small_model_cpu_dir = serialize_model(small_model, 'small_model-cpu')
large_model_dir = serialize_model(large_model, 'large_model')
large_model_cpu_dir = serialize_model(large_model, 'large_model-cpu')