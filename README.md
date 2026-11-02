# AI-Based Intrusion Detection System

A Python project that classifies network traffic as normal or attack using
machine learning and a neural network.

## What it does
- Explores the NSL-KDD dataset (distributions, correlation heatmap, attack types)
- Trains Random Forest and SVC models
- Trains a neural network in PyTorch
- Compares all three models on the test set
- Includes a socket server and client that simulate real-time detection

## Dataset
NSL-KDD (KDDTrain+ and KDDTest+), included in the `data/` folder.
Source: https://www.unb.ca/cic/datasets/nsl.html

## How to Run
Run the steps in order.

1. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn torch joblib
   ```
2. Exploratory data analysis: `python eda.py`
3. Train and save the ML models (Random Forest, SVC): `python ml_models.py`
4. Train the neural network: `python nn_models.py`
5. Compare the models: `python compare_models.py`
6. Start the server and wait until it says it is listening: `python server.py`
7. In a second terminal, run the client: `python client.py`

The client sends sample records to the server, which returns the predicted
class (`0` = normal, `1` = attack) and its probability. The server must be
running before the client starts.

## Note
NSL-KDD is an older dataset, and the socket demo uses test records rather than live traffic.
