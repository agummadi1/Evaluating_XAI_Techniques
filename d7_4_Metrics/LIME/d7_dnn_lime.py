import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Start timing
start_time = time.time()

# Combine feature columns with the label column
req_cols = [
'HH_L3_radius',
'HH_L5_radius',
'HH_L3_covariance',
'HH_L1_mean',
'HH_L3_mean',
'HH_L5_mean',
'HH_L1_radius',
'HH_L5_magnitude',
'HH_L3_weight',
'HH_L5_weight',
'HH_L5_covariance',
'HH_L3_magnitude',
'HH_L3_std',
'HH_L1_std',
'HH_L5_std',
'HpHp_L0.01_pcc',
'HH_L5_pcc',
'HH_L3_pcc',
'HH_L1_weight',
'HH_L1_magnitude',
'label']
num_columns = 20
num_labels = 6

fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)

# Model Parameters
batch_size = 32
epochs = 10

# XAI Samples
samples = 2500


# Specify the name of the output text file
output_file_name = "d7_DNN_LIME_output.txt"
with open(output_file_name, "a") as f:
    print('--------------------------------------------------', file=f)
    print('start', file=f)

print('--------------------------------------------------')
print('DNN')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Load your dataset
df = pd.read_csv('device7_top_20_features.csv', usecols=req_cols)

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Map labels from 1 to 6 to 0 to 5
y -= 1

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(num_columns,)),
    Dense(64, activation='relu'),
    Dense(num_labels, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

# Calculate accuracy
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

# Create a function to predict probabilities using the trained model
def predict_proba(X):
    return model.predict(X)

# Create a Lime explainer object
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values, class_names=np.unique(y_train),
                                 discretize_continuous=True)

# Generate Lime explanations for each sample in the test set
for i in range(samples):
    exp = explainer.explain_instance(X_test.iloc[i].values, predict_proba, num_features=num_columns,
                                     top_labels=num_labels)
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
    exp = explainer.explain_instance(X_test.iloc[i].values, predict_proba, num_features=num_columns,
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

# Removed sparsity calculation

with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('DNN', file=f)
    print('--------------------------------------------------', file=f)
    print('Accuracy:', accuracy, file=f)
    print('Samples:', samples, file=f)

end_time = time.time()

execution_time = end_time - start_time

execution_time_hours = execution_time / 3600

print("Execution time: %s hours" % execution_time_hours)

with open(output_file_name, "a") as f:
    print('Execution time:', execution_time_hours, 'hours', file=f)
