# compare_models.py
from joblib import load
import torch
import numpy as np
from utils import load_data, load_preprocessor, preprocess_df, save_metrics_table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load preprocessor and data
preprocessor = load_preprocessor('models/preprocessor.joblib')
train_df, test_df = load_data('data/KDDTrain.csv', 'data/KDDTest.csv')
X_test, y_test = preprocess_df(test_df, preprocessor)

# Load classical models
rf = load('models/rf_model.joblib')
svc = load('models/svc_model.joblib')

# RF
rf_pred = rf.predict(X_test)
# SVC
svc_pred = svc.predict(X_test)

# NN
from nn_model import Net  # careful: this will import training code if not modularized; alternatively recreate Net here
# Safer to recreate Net class locally:
import json
with open('models/nn_meta.json', 'r') as f:
    meta = json.load(f)
input_dim = X_test.shape[1]
num_classes = meta.get('num_classes', 2)

class NetLocal(torch.nn.Module):
    def __init__(self, input_dim):
        super(NetLocal, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = NetLocal(input_dim).to(device)
nn_model.load_state_dict(torch.load('models/nn_model.pth', map_location=device))
nn_model.eval()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    nn_out = nn_model(X_test_tensor)
    nn_pred = nn_out.argmax(dim=1).cpu().numpy()

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

metrics = {
    'RandomForest': compute_metrics(y_test, rf_pred),
    'SVC': compute_metrics(y_test, svc_pred),
    'NeuralNetwork': compute_metrics(y_test, nn_pred)
}

print("Metrics summary:\n", metrics)
save_metrics_table(metrics, filename='models/metrics_comparison.csv')
print("Saved metrics to models/metrics_comparison.csv")
