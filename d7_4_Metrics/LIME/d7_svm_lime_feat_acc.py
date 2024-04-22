import time
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc

# Start timing
start = time.time()

# Combine feature columns with the label column
req_cols = [
    'HH_L3_radius', 'HH_L3_magnitude', 'HH_L5_magnitude', 'HH_L1_mean', 'HH_L3_mean',
    'HH_L1_radius', 'HpHp_L0.01_pcc', 'HH_L1_magnitude', 'HH_L3_weight', 'HH_L5_radius',
    'HH_L5_std', 'HH_L5_weight', 'HH_L5_pcc', 'HH_L3_pcc', 'HH_L5_covariance',
    'HH_L5_mean', 'HH_L3_covariance', 'HH_L3_std', 'HH_L1_std', 'HH_L1_weight', 'label'
]
num_columns = 21  # 20 features + 1 label

# Model Parameters
max_iter = 10
loss = 'log'  # Change the loss function to 'log'
gamma = 0.1
split = 0.7

# XAI Samples
samples = 1000

# Specify the name of the output text file
output_file_name = "d7_svm_lime_desc_output.txt"
with open(output_file_name, "a") as f:
    print('---------------------------------------------------------------------------------', file=f)
    print('LIME', file=f)
    print('---------------------------------------------------------------------------------', file=f)

# Define function to calculate accuracy
def calculate_accuracy(features_count):
    print('---------------------------------------------------------------------------------')
    print(f'Training SVM with last {features_count} features')
    print('---------------------------------------------------------------------------------')

    # Load dataset
    df = pd.read_csv('device7_top_20_features.csv', usecols=req_cols[-features_count:])

    # Define features and labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)

    # Define SVM model
    rbf_feature = RBFSampler(n_components=100, gamma=gamma, random_state=1)  # Ensure 100 features for RBFSampler
    X_features = rbf_feature.fit_transform(X_train)
    clf = SGDClassifier(max_iter=max_iter, loss=loss)
    clf.fit(X_features, y_train)

    # Model prediction
    X_test_rbf = rbf_feature.transform(X_test)  # Transform test data using RBFSampler
    rbf_pred = clf.predict(X_test_rbf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, rbf_pred)
    return accuracy

# Calculate accuracy for each case (15, 10, 5 features) and print to file
for features_count in [20, 15, 10, 5]:
    accuracy = calculate_accuracy(features_count)
    with open(output_file_name, "a") as f:
        print(f'\nAccuracy with last {features_count} features:', accuracy, file=f)

# End timing
end_time = time.time()

# Calculate execution time
execution_time = (end_time - start) / 3600

# Write total execution time to file
with open(output_file_name, "a") as f:
    print('\nTime Taken for Running full(in hours):', execution_time, file=f)
