# server.py
import socket
import json
import pandas as pd
from joblib import load
import traceback

HOST = 'localhost'
PORT = 9999

# Load preprocessor, model and feature columns
preprocessor = load('models/preprocessor.joblib')
with open('models/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)
num_cols = feature_cols['num_cols']
cat_cols = feature_cols['cat_cols']
feature_order = num_cols + cat_cols  # columns expected in incoming JSON

# load model (choose RF by default)
from joblib import load as jl_load
model = jl_load('models/rf_model.joblib')

def handle_request(conn):
    try:
        data = conn.recv(65536)
        if not data:
            return
        payload = json.loads(data.decode('utf-8'))
        # Expect either a dict of {col: value} or list (must be list with same order)
        if isinstance(payload, dict):
            # make sure all expected features present
            row = {k: payload.get(k, None) for k in feature_order}
            df = pd.DataFrame([row], columns=feature_order)
        elif isinstance(payload, list):
            # interpret as values in feature_order order
            df = pd.DataFrame([payload], columns=feature_order)
        else:
            resp = {'error': 'Invalid payload format. Send dict or list.'}
            conn.send(json.dumps(resp).encode())
            return

        # Preprocess
        X_transformed = preprocessor.transform(df)  # returns numpy array
        pred = model.predict(X_transformed)[0]
        # If model supports predict_proba:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_transformed)[0].max())

        resp = {'prediction': int(pred), 'probability': proba}
        conn.send(json.dumps(resp).encode())
    except Exception:
        tb = traceback.format_exc()
        conn.send(json.dumps({'error': 'server error', 'trace': tb}).encode())

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                handle_request(conn)

if __name__ == '__main__':
    start_server()
