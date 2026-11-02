# nn_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import load, dump
from utils import load_data, load_preprocessor, preprocess_df
import numpy as np
import os

os.makedirs('models', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data and preprocessor
train_df, test_df = load_data('data/KDDTrain.csv', 'data/KDDTest.csv')
preprocessor = load_preprocessor('models/preprocessor.joblib')

X_train, y_train = preprocess_df(train_df, preprocessor)
X_test, y_test = preprocess_df(test_df, preprocessor)

# Convert to tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long)

# Create dataset and split into train/val
dataset = TensorDataset(X_tensor, y_tensor)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

input_dim = X_train.shape[1]
num_classes = 2

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = Net(input_dim).to(device)

# compute class weights to address imbalance
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
best_val_f1 = 0.0
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())
    val_acc = accuracy_score(all_true, all_preds)
    val_f1 = f1_score(all_true, all_preds, zero_division=0)
    print(f"Epoch {epoch}/{num_epochs} — loss: {epoch_loss:.4f} val_acc: {val_acc:.4f} val_f1: {val_f1:.4f}")

    # Save best
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'models/nn_model.pth')

# Evaluate on test
model.load_state_dict(torch.load('models/nn_model.pth', map_location=device))
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    out_test = model(X_test_tensor)
    preds_test = out_test.argmax(dim=1).cpu().numpy()

print("Neural Network test report:\n", classification_report(y_test, preds_test))

# Save the final model state dict already done, but also save a small metadata file
import json
meta = {'input_dim': input_dim, 'num_classes': num_classes}
with open('models/nn_meta.json', 'w') as f:
    json.dump(meta, f)

print("Saved NN model to models/nn_model.pth and metadata to models/nn_meta.json")
