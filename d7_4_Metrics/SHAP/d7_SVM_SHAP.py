###################################################
#               Parameter Setting                #
###################################################

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

# Model Parameters
max_iter = 10
loss = 'log'  # Change the loss function to 'log'
gamma = 0.1

# XAI Samples
samples = 1000

# Specify the name of the output text file
output_file_name = "SVM_SHAP_output.txt"
with open(output_file_name, "w") as f:
    print('---------------------------------------------------------------------------------', file=f)
    print('SVM', file=f)
    print('---------------------------------------------------------------------------------', file=f)

###################################################
###################################################
###################################################
###################################################

print('---------------------------------------------------------------------------------')
print('SVM')
print('---------------------------------------------------------------------------------')


print('---------------------------------------------------------------------------------')
print('Importing Libraries')
print('---------------------------------------------------------------------------------')


import time
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc

# Start timing
start = time.time()

# Combine feature columns with the label column
req_cols = [
'HH_L3_radius',
'HH_L3_magnitude',
'HH_L5_magnitude',
'HH_L1_mean',
'HH_L3_mean',
'HH_L1_radius',
'HpHp_L0.01_pcc',
'HH_L1_magnitude',
'HH_L3_weight',
'HH_L5_radius',
'HH_L5_std',
'HH_L5_weight',
'HH_L5_pcc',
'HH_L3_pcc',
'HH_L5_covariance',
'HH_L5_mean',
'HH_L3_covariance',
'HH_L3_std',
'HH_L1_std',
'HH_L1_weight',
'label']
num_columns = 21  # 20 features + 1 label

#----------------------------------------

print('---------------------------------------------------------------------------------')
print('Defining features')
print('---------------------------------------------------------------------------------')

# Load dataset
df = pd.read_csv('device7_top_20_features.csv', usecols=req_cols)

# Define features and labels
X = df.drop(columns=['label'])
y = df['label']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split)

print('---------------------------------------------------------------------------------')
print('Model training')
print('---------------------------------------------------------------------------------')

# Define SVM model
rbf_feature = RBFSampler(n_components=100, gamma=gamma, random_state=1)  # Ensure 100 features for RBFSampler
X_features = rbf_feature.fit_transform(X_train)
clf = SGDClassifier(max_iter=max_iter, loss=loss)
clf.fit(X_features, y_train)

# Model training time
end = time.time()
training_time = (end - start) / 3600  # Time taken for model training in hours

print('---------------------------------------------------------------------------------')
print('Model Prediction')
print('---------------------------------------------------------------------------------')

# Model prediction time
start = time.time()
X_test_rbf = rbf_feature.transform(X_test)  # Transform test data using RBFSampler
rbf_pred = clf.predict(X_test_rbf)
end = time.time()
prediction_time = (end - start) / 3600  # Time taken for model prediction in hours

print('---------------------------------------------------------------------------------')
print('Metrics')
print('---------------------------------------------------------------------------------')

# Calculate accuracy
accuracy = accuracy_score(y_test, rbf_pred)
print('Accuracy:', accuracy)

# Generate SHAP explanation
print('---------------------------------------------------------------------------------')
print('Generating SHAP explanation')
print('---------------------------------------------------------------------------------')
test = X_test
train = X_train
start_index = 0
end_index = samples

# Create a wrapper function for prediction probabilities
predict_proba_fn = lambda x: clf.predict_proba(x)

# Create the KernelExplainer using the wrapper function and transformed test data
explainer = shap.KernelExplainer(predict_proba_fn, X_test_rbf[start_index:end_index])
shap_values = explainer.shap_values(X_test_rbf[start_index:end_index])

# Generate feature importance
vals = np.abs(shap_values).mean(1)
feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name', 'feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

# End timing
end_time = time.time()

# Calculate execution time
execution_time = (end_time - start) / 3600

# Write outputs to file
with open(output_file_name, "a") as f:
    print('\nSamples:', samples, file=f) 
    print('\nAccuracy:', accuracy, file=f)
    print('\nTime Taken for Running training(in hours):', training_time, file=f)
    print('\nTime Taken for Running prediction(in hours):', prediction_time, file=f)
    print('\nTime Taken for Running full(in hours):', execution_time, file=f)
    print('\nFeature Importance (Descending Order):', file=f)
    print(feature_importance, file=f)
