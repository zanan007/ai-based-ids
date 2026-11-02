# client.py
import socket
import json
import pandas as pd
import random

HOST = 'localhost'
PORT = 9999

# Load raw test file so we can pick real examples (raw, unscaled)
df = pd.read_csv('data/KDDTest.csv')

# Remove label column to form payload features
feature_df = df.drop(columns=['label']).reset_index(drop=True)

def send_sample(sample_series):
    payload = sample_series.to_dict()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.send(json.dumps(payload).encode('utf-8'))
        data = s.recv(65536)
    resp = json.loads(data.decode('utf-8'))
    return resp

if __name__ == '__main__':
    for _ in range(5):
        idx = random.randint(0, len(feature_df)-1)
        sample = feature_df.loc[idx]
        res = send_sample(sample)
        print(f"Index {idx} → Server response: {res}")
