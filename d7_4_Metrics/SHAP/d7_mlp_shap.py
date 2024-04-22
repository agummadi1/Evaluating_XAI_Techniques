import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import shap
import time

# Start timing
start_time = time.time()

# Combine feature columns with the label column
req_cols = [
    'HH_L3_pcc',
    'HH_L5_pcc',
    'HH_L3_std',
    'HH_L1_weight',
    'HH_L5_std',
    'HH_L1_radius',
    'HH_L5_weight',
    'HH_L1_magnitude',
    'HH_L1_std',
    'HH_L5_magnitude',
    'HH_L5_covariance',
    'HH_L3_weight',
    'HH_L3_magnitude',
    'HH_L1_mean',
    'HH_L3_covariance',
    'HH_L3_mean',
    'HH_L3_radius',
    'HpHp_L0.01_pcc',
    'HH_L5_radius',
    'HH_L5_mean',
    'label'
]
num_columns = 20  # 20 features
num_labels = 6  # 6 labels

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

# Model Parameters
batch_size = 32
epochs = 10

# XAI Samples
samples = 1000

# Specify the name of the output text file
output_file_name = "d7_MLP_SHAP_output.txt"

print('--------------------------------------------------')
print('MLP')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Load your dataset
df = pd.read_csv('week8/top20_datasets/device7_top_20_features.csv', usecols=req_cols)

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

# Define the model
mlp = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', batch_size=batch_size,
                    max_iter=epochs, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, mlp.predict(X_test))

# Create an explainer object
explainer = shap.KernelExplainer(mlp.predict_proba, X_train.iloc[:100].values)

# Generate SHAP explanations for each sample in the test set
shap_values = explainer.shap_values(X_test.values)

# Calculate the mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# Sort and print feature importance
zipped_lists = list(zip(X_train.columns.values, mean_abs_shap_values))
zipped_lists.sort(key=lambda x: x[1].mean(), reverse=True)

sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

print('Feature Importance (Descending Order):')
for k, v in zip(sorted_list1, sorted_list2):
    print(k, v.mean())

# Generate sparsity graph
thresholds = [i / 10 for i in range(11)]
sparsity_values = []

for threshold in thresholds:
    count_below_threshold = sum(1 for value in mean_abs_shap_values if (value < threshold).any())
    sparsity_values.append(count_below_threshold / len(mean_abs_shap_values))

print('Sparsity:', sparsity_values)

# Save sparsity graph
plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')
plt.title('Sparsity vs. Threshold')
plt.savefig('d7_sparsity_MLP_SHAP.png')
plt.clf()

# Write results to output file
with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('MLP', file=f)
    print('--------------------------------------------------', file=f)
    print('Feature Importance (Descending Order):', file=f)
    for k, v in zip(sorted_list1, sorted_list2):
        print(k, v.mean(), file=f)
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
