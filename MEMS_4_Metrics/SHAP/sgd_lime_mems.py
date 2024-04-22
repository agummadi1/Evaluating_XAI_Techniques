import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import time

# Start timing
start_time = time.time()

# Define constants or variables
req_cols = ['x', 'label']  # For Descriptive accuracy, remove one feature after each run and re-run
num_columns = 3  # Fill in the number of columns in your dataset

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

# Model Parameters
max_iter = 1000
tol = 1e-3

# XAI Samples
samples = 1000

# Specify the name of the output text file
output_file_name = "SGD_LIME_output.txt"

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

# Create an explainer object with original feature names
explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns.tolist(),
                                 class_names=np.unique(y_train), discretize_continuous=False)

# Initialize feature_val
feature_val = [0] * len(X_train.columns)

# Generate Lime explanations for each sample in the test set
for i in range(samples):
    exp = explainer.explain_instance(X_test.iloc[i].values, sgd.predict_proba, num_features=num_columns,
                                     top_labels=num_columns)
    lime_list = exp.as_list()
    lime_list.sort()
    print(f"Lime explanations for sample {i}: {lime_list}")

    for item in lime_list:
        feature_name = item[0]
        if feature_name in X_train.columns:
            feature_index = X_train.columns.get_loc(feature_name)
            feature_val[feature_index] += abs(item[1])
        else:
            print(f"Feature '{feature_name}' not found in X_train columns.")

    print('Progress:', 100 * (i + 1) / samples, '%')

# Calculate feature importance based on Lime explanations
feature_val = [x / samples for x in feature_val]  # Divide by the number of samples

# Sort and print feature importance
zipped_lists = list(zip(X_train.columns.values, feature_val))  # Using X_train.columns.values to get feature names
zipped_lists.sort(key=lambda x: x[1], reverse=True)

sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

print('Feature Importance (Descending Order):')
for k, v in zip(sorted_list1, sorted_list2):
    print(k, v)

# Generate sparsity graph
thresholds = [i / 10 for i in range(11)]
sparsity_values = []

for threshold in thresholds:
    count_below_threshold = sum(1 for value in feature_val if value < threshold)
    sparsity_values.append(count_below_threshold / len(feature_val))

print('Sparsity:', sparsity_values)

# Save sparsity graph
plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')
plt.title('Sparsity vs. Threshold')
plt.savefig('sparsity_LIME.png')
plt.clf()

# Write results to output file
with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('SGD', file=f)
    print('--------------------------------------------------', file=f)
    print('Feature Importance (Descending Order):', file=f)
    for k, v in zip(sorted_list1, sorted_list2):
        print(k, v, file=f)
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
