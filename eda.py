# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="whitegrid")

train = pd.read_csv('data/KDDTrain.csv')
test = pd.read_csv('data/KDDTest.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain summary statistics:\n", train.describe(include='all'))


print("\nUnique values in attack_type column:")
print(train['attack_type'].unique())

train['binary_label'] = train['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
test['binary_label'] = test['attack_type'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Percentage distribution of normal vs attack (train)
dist_train = train['binary_label'].value_counts(normalize=True) * 100
print("\nTrain binary distribution (percent):\n", dist_train)
plt.figure(figsize=(6,4))
dist_train.plot(kind='bar')
plt.title('Train: Normal vs Attack (%)')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('train_label_dist.png')
plt.show()
plt.close()

# Test binary distribution
dist_test = test['binary_label'].value_counts(normalize=True) * 100
print("\nTest binary distribution (percent):\n", dist_test)
plt.figure(figsize=(6,4))
dist_test.plot(kind='bar')
plt.title('Test: Normal vs Attack (%)')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('test_label_dist.png')
plt.show()
plt.close()

# Attack type distribution in training dataset (percentage)

attack_types = train.loc[train['binary_label'] == 'attack', 'attack_type']
attack_pct = attack_types.value_counts(normalize=True) * 100
print("\nAttack type distribution:\n", attack_pct)
plt.figure(figsize=(10,6))
attack_pct.sort_values(ascending=False).plot(kind='bar')
plt.title('Attack Type Distribution in Train (%)')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('attack_type_distribution.png')
plt.show()
plt.close()

# Correlation heatmap for numeric features
numeric_train = train.select_dtypes(include=[np.number])
corr = numeric_train.corr()
plt.figure(figsize=(14,12))
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.1)
plt.title('Feature Correlation Heatmap (numeric features)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()

print("EDA plots saved: train_label_dist.png, test_label_dist.png, attack_type_distribution.png, correlation_heatmap.png")