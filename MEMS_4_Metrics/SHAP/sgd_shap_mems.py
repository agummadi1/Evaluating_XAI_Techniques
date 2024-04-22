import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import shap
import time

# Start timing
start_time = time.time()

# Define constants or variables
req_cols = ['x', 'y', 'label']  # For Descriptive accuracy, remove one feature after each run and re-run
num_columns = 3  # Fill in the number of columns in your dataset

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

#Model Parameters
max_iter = 1000
tol = 1e-3

# XAI Samples
samples = 1000

# Specify the name of the output text file
output_file_name = "SGD_SHAP_output.txt"

print('--------------------------------------------------')
print('SGD')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Load your dataset
df = pd.read_csv('mems_dataset.csv', usecols=req_cols)

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

# Define the model
sgd = SGDClassifier(loss='log', max_iter=max_iter, tol=tol)

# Train the model
sgd.fit(X_train, y_train)

# Calculate accuracy
y_pred = sgd.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Create an explainer object
explainer = shap.KernelExplainer(sgd.predict_proba, X_train)

# Generate SHAP explanations for each sample in the test set
shap_values = explainer.shap_values(X_test)

# Calculate the mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values[1]), axis=0)

# Sort and print feature importance
zipped_lists = list(zip(X_train.columns.values, mean_abs_shap_values))
zipped_lists.sort(key=lambda x: np.mean(x[1]), reverse=True)

sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

print('Feature Importance (Descending Order):')
for k, v in zip(sorted_list1, sorted_list2):
    print(k, np.mean(v))

# Generate sparsity graph
thresholds = [i / 10 for i in range(11)]
sparsity_values = []

for threshold in thresholds:
    count_below_threshold = sum(1 for value in mean_abs_shap_values if value < threshold)
    sparsity_values.append(count_below_threshold / len(mean_abs_shap_values))

print('Sparsity:', sparsity_values)

# Save sparsity graph
plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')
plt.title('Sparsity vs. Threshold')
plt.savefig('sparsity_SGD_SHAP.png')
plt.clf()

# Write results to output file
with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('SGD', file=f)
    print('--------------------------------------------------', file=f)
    print('Feature Importance (Descending Order):', file=f)
    for k, v in zip(sorted_list1, sorted_list2):
        print(k, np.mean(v), file=f)
    print('Accuracy:', accuracy, file=f)
    print('Sparsity:', sparsity_values, file=f)
    print('Samples:', samples, file=f)

# End timing
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time

# Calculate execution time in hours
execution_time_hours = execution_time / 3600  # 3600 seconds in an hour

# Print execution time in hours
print("Execution time: %s hours" % execution_time_hours)

# Write execution time to output file in hours
with open(output_file_name, "a") as f:
    print('Execution time:', execution_time_hours, 'hours', file=f)