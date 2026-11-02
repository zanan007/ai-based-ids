import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load
import os
import json

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def binary_label(y_series):
    """Dataset already has 0 and 1 labels. Return as-is."""
    return y_series.astype(int)

def build_and_fit_preprocessor(df_train, save_path=os.path.join(MODELS_DIR, 'preprocessor.joblib')):

    # Identify columns
    num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

    # Remove label and attack_type (not features)
    for col in ['label', 'attack_type']:
        if col in num_cols:
            num_cols.remove(col)
        if col in cat_cols:
            cat_cols.remove(col)

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    # Fit preprocessor (exclude label + attack_type)
    df_features = df_train.drop(columns=['label', 'attack_type'])
    preprocessor.fit(df_features)

    dump(preprocessor, save_path)

    feature_columns = {
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_columns, f)

    return preprocessor, feature_columns

def load_preprocessor(path=os.path.join(MODELS_DIR, 'preprocessor.joblib')):
    return load(path)

def preprocess_df(df, preprocessor):

    df_copy = df.copy()

    # y = binary labels (already 0 or 1)
    if 'label' in df_copy.columns:
        y = df_copy['label'].astype(int)
        X_df = df_copy.drop(columns=['label', 'attack_type'])
    else:
        y = None
        X_df = df_copy

    X_transformed = preprocessor.transform(X_df)
    return X_transformed, y

def save_metrics_table(metrics_dict, filename='models/metrics_summary.csv'):
    import csv
    keys = list(next(iter(metrics_dict.values())).keys())
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model'] + keys)
        for model, vals in metrics_dict.items():
            writer.writerow([model] + [vals[k] for k in keys])
