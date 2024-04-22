import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import time

# Start timing
start_time = time.time()

# Combine feature columns with the label column
req_cols = [
'HH_L5_radius',
'HH_L5_std',
'HH_L1_weight',
'HH_L5_covariance',
'HH_L3_weight',
'HH_L3_pcc',
'HH_L5_weight',
'HH_L5_magnitude',
'HH_L5_mean',
'HH_L1_radius',
'HH_L3_radius',
'HH_L3_covariance',
'HH_L5_pcc',
'HpHp_L0.01_pcc',
'HH_L3_magnitude',
'HH_L3_std',
'HH_L1_magnitude',
'HH_L3_mean',
'HH_L1_mean',
'HH_L1_std',
'label']
num_columns = 21  # 20 features + 1 label

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

# Model Parameters
hidden_layer_sizes = (100,)  # Single hidden layer with 100 neurons
activation = 'relu'  # Activation function for hidden layers
solver = 'adam'  # Optimization solver
alpha = 0.0001  # L2 penalty (regularization term) parameter
batch_size = 'auto'  # Size of minibatches for stochastic optimizers
learning_rate = 'constant'  # Learning rate schedule for weight updates
learning_rate_init = 0.001  # Initial learning rate
max_iter = 200  # Maximum number of iterations
shuffle = True  # Whether to shuffle samples in each iteration
random_state = 42  # Random seed for weight initialization

# XAI Samples
samples = 2500

# Specify the name of the output text file
output_file_name = "d7_MLP_LIME_output.txt"
with open(output_file_name, "a") as f:
    print('--------------------------------------------------', file=f)
    print('start', file=f)

print('--------------------------------------------------')
print('Multi-Layer Perceptron (MLP)')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Load your dataset
df = pd.read_csv('device7_top_20_features.csv', usecols=req_cols)

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

# Define the model
mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                               alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                               learning_rate_init=learning_rate_init, max_iter=max_iter, shuffle=shuffle,
                               random_state=random_state)

# Train the model
mlp_classifier.fit(X_train, y_train)

# Calculate accuracy
y_pred = mlp_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Create a Lime explainer object
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values, class_names=np.unique(y_train),
                                 discretize_continuous=True)

# Generate Lime explanations for each sample in the test set
for i in range(samples):
    exp = explainer.explain_instance(X_test.iloc[i].values, mlp_classifier.predict_proba, num_features=num_columns,
                                     top_labels=num_columns)
    lime_list = exp.as_list()
    lime_list.sort()
    print(lime_list)

    print('Progress:', 100 * (i + 1) / samples, '%')

print('---------------------------------------------------------------------------------')
print('Generating Explainer')
print('---------------------------------------------------------------------------------')

start = time.time()

explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=list(X_train.columns.values),
                                                  class_names=np.unique(y_train), discretize_continuous=True)

feat_list = req_cols[:-1]

feat_dict = dict.fromkeys(feat_list, 0)

c = 0

num_columns = df.shape[1] - 1
feature_name = req_cols[:-1]
feature_name.sort()
feature_val = []

for i in range(0, num_columns):
    feature_val.append(0)

for i in range(0, samples):
    exp = explainer.explain_instance(X_test.iloc[i].values, mlp_classifier.predict_proba, num_features=num_columns,
                                     top_labels=num_columns)

    lime_list = exp.as_list()
    lime_list.sort()
    print(lime_list)

    for j in range(0, num_columns):
        feature_val[j] += abs(lime_list[j][1])

    c = c + 1
    print('progress', 100 * (c / samples), '%')

divider = samples
feature_val = [x / divider for x in feature_val]

zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1], reverse=True)

sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

print('----------------------------------------------------------------------------------------------------------------')

for item1, item2 in zip(sorted_list1, sorted_list2):
    print(item1, item2)
    with open(output_file_name, "a") as f:
        print(item1, item2, file=f)

for k in sorted_list1:
    with open(output_file_name, "a") as f:
        print("df.pop('", k, "')", sep='', file=f)

with open(output_file_name, "a") as f:
    print("Trial_ =[", file=f)
for k in sorted_list1:
    with open(output_file_name, "a") as f:
        print("'", k, "',", sep='', file=f)
with open(output_file_name, "a") as f:
    print("]", file=f)

print('---------------------------------------------------------------------------------')

end = time.time()
with open(output_file_name, "a") as f:
    print('ELAPSE TIME LIME GLOBAL: ', (end - start) / 60, 'min', file=f)
print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating Sparsity Graph')
print('---------------------------------------------------------------------------------')
print('')

# Calculate normalized list
min_value = min(feature_val)
max_value = max(feature_val)
normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]


Spar = []
X_axis = []

# Corrected logic for calculating Sparsity and Spar
Spar = []
for i in range(0, 11):
    threshold = i / 10
    count_below_threshold = sum(1 for value in normalized_list if value < threshold)
    Sparsity = count_below_threshold / len(normalized_list)
    Spar.append(Sparsity)
    X_axis.append(threshold)
    print('Progress:', 100 * (i + 1) / 11, '%')

with open(output_file_name, "a") as f:
    print('y_axis_RF = ', Spar, '', file=f)
with open(output_file_name, "a") as f:
    print('x_axis_RF = ', X_axis, '', file=f)

plt.plot(X_axis, Spar, marker='o', linestyle='-')

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.title('Values vs. X-Axis')

plt.savefig('d7_sparsity_RF_LIME.png')
plt.clf()

Sparsity = count_below_threshold / len(normalized_list)
print('Sparsity = ', Sparsity)

feature_val = [x / samples for x in feature_val]

zipped_lists = list(zip(X_train.columns.values, feature_val))
zipped_lists.sort(key=lambda x: x[1], reverse=True)

sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

print('Feature Importance (Descending Order):')
for k, v in zip(sorted_list1, sorted_list2):
    print(k, v)

thresholds = [i / 10 for i in range(11)]
sparsity_values = []

for threshold in thresholds:
    count_below_threshold = sum(1 for value in feature_val if value < threshold)
    sparsity_values.append(count_below_threshold / len(feature_val))

print('Sparsity:', sparsity_values)

plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')
plt.title('Sparsity vs. Threshold')
plt.savefig('d7_sparsity_MLP_LIME.png')
plt.clf()

with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('Multi-Layer Perceptron (MLP)', file=f)
    print('--------------------------------------------------', file=f)
    print('Feature Importance (Descending Order):', file=f)
    for k, v in zip(sorted_list1, sorted_list2):
        print(k, v, file=f)
    print('Accuracy:', accuracy, file=f)
    print('Sparsity:', sparsity_values, file=f)
    print('Samples:', samples, file=f)

end_time = time.time()

execution_time = end_time - start_time

execution_time_hours = execution_time / 3600

print("Execution time: %s hours" % execution_time_hours)

with open(output_file_name, "a") as f:
    print('Execution time:', execution_time_hours, 'hours', file=f)
