# ml_models.py
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils import load_data, build_and_fit_preprocessor, preprocess_df, save_metrics_table
import numpy as np

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# 1) Load raw csv

train_df, test_df = load_data('data/KDDTrain.csv', 'data/KDDTest.csv')

# 2) Build and save preprocessor

preprocessor, feature_cols = build_and_fit_preprocessor(train_df)


# 3) Preprocess datasets

X_train, y_train = preprocess_df(train_df, preprocessor)
X_test, y_test = preprocess_df(test_df, preprocessor)


# Debug: check unique labels

print(train_df['label'].unique())
print(train_df['label'].value_counts())

print("Unique labels in y_train:", np.unique(y_train))
print("Unique labels in y_test:", np.unique(y_test))
print("Counts in y_train:", np.bincount(y_train))
print("Counts in y_test:", np.bincount(y_test))

if len(np.unique(y_train)) < 2:
    raise ValueError("y_train has less than 2 classes. Check the binary_label function and dataset.")

# 4) Train RandomForest and SVC

print("Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
dump(rf, 'models/rf_model.joblib')

print("Training SVC (may take a while)...")
svc = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42
)
svc.fit(X_train, y_train)
dump(svc, 'models/svc_model.joblib')

# 5) Evaluate models

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

rf_pred = rf.predict(X_test)
svc_pred = svc.predict(X_test)

metrics = {
    'RandomForest': compute_metrics(y_test, rf_pred),
    'SVC': compute_metrics(y_test, svc_pred)
}

print("RandomForest classification report:\n", classification_report(y_test, rf_pred))
print("SVC classification report:\n", classification_report(y_test, svc_pred))

# 6) Save metrics CSV

save_metrics_table(metrics, filename='models/metrics_summary.csv')
print("Saved models and metrics to /models")
