
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import time
 
# Start timing
start_time = time.time()
 
# Combine feature columns with the label column
req_cols = [ 
# 'HH_L3_covariance',
# 'HH_L5_weight',
# 'HH_L5_mean',
# 'HH_L5_std',
# 'HH_L5_magnitude',
# 'HH_L5_radius',
# 'HH_L5_covariance',
# 'HH_L5_pcc',
# 'HH_L3_weight',
# 'HH_L3_mean',
# 'HH_L3_std',
# 'HH_L3_magnitude',
# 'HH_L3_radius',
# 'HH_L3_pcc',
# 'HpHp_L0.01_pcc',
'HH_L1_weight',
'HH_L1_mean',
'HH_L1_std',
'HH_L1_magnitude',
'HH_L1_radius',
'label']
num_columns = 21  # 20 features + 1 label
 
fraction = 0.5  # how much of that database you want to use
frac_normal = 0.2  # how much of the normal classification you want to reduce
split = 0.70  # how you want to split the train/test data (this is percentage for train)
 
# Model Parameters
max_depth = 5
min_samples_split = 2
 
# XAI Samples
samples = 1000
 
# Specify the name of the output text file
output_file_name = "d7_DT_LIME_output.txt"
with open(output_file_name, "a") as f:print('start',file = f)
 
print('--------------------------------------------------')
print('Decision Tree')
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
decision_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

# Train the model
decision_tree.fit(X_train, y_train)
DecisionTreeClassifier(max_depth=5)
 
# Calculate accuracy
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

Acc = accuracy_score(y_test, y_pred)
Precision = precision_score(y_test, y_pred, average='macro')
Recall = recall_score(y_test, y_pred, average='macro')
F1 =  f1_score(y_test, y_pred, average='macro')
BACC = balanced_accuracy_score(y_test, y_pred)
MCC = matthews_corrcoef(y_test, y_pred)

# with open(output_file_name, "a") as f:print('Accuracy total: ', Acc,file=f)
print('Accuracy total: ', Acc)
print('Precision total: ', Precision )
print('Recall total: ', Recall )
print('F1 total: ', F1 )
print('BACC total: ', BACC)
print('MCC total: ', MCC)

# Create an explainer object
explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns.values,
                                 class_names=np.unique(y_train), discretize_continuous=True)
 
# Generate Lime explanations for each sample in the test set
for i in range(samples):
    exp = explainer.explain_instance(X_test.iloc[i].values, decision_tree.predict_proba, num_features=num_columns,
                                     top_labels=num_columns)
    lime_list = exp.as_list()
    lime_list.sort()
    print(lime_list)
    for item in lime_list:
        feature_name = item[0]
        if feature_name in X_train.columns:
            feature_index = X_train.columns.get_loc(feature_name)
            feature_val[feature_index] += abs(item[1])
        else:
            print(f"Feature '{feature_name}' not found in X_train columns.")
 
    print('Progress:', 100 * (i + 1) / samples, '%')
 
import lime
print('---------------------------------------------------------------------------------')
print('Generating Explainer')
print('---------------------------------------------------------------------------------')



# test.pop ('Label')
print('------------------------------------------------------------------------------')

#START TIMER MODEL
start = time.time()

# test2 = test
# test = test.to_numpy()

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names= list(X_train.columns.values) , class_names=np.unique(y_train) , discretize_continuous=True)
# explainer = LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns.values,
#                                  class_names=np.unique(y_train), discretize_continuous=True)

#creating dict 
feat_list = req_cols[:-1]
# print(feat_list)

feat_dict = dict.fromkeys(feat_list, 0)
# print(feat_dict)
c = 0

num_columns = df.shape[1] - 1
feature_name = req_cols[:-1]
feature_name.sort()
# print('lista',feature_name)
feature_val = []

for i in range(0,num_columns): feature_val.append(0)

for i in range(0,samples):

# i = sample
    # exp = explainer.explain_instance(test[i], rf.predict_proba)
    
    exp = explainer.explain_instance(X_test.iloc[i].values, decision_tree.predict_proba, num_features=num_columns, top_labels=num_columns)
    # exp.show_in_notebook(show_table=True, show_all=True)
    
    #lime list to string
    lime_list = exp.as_list()
    lime_list.sort()
    print(lime_list)
    for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    # print ('debug here',lime_list[1][1])

    # lime_str = ' '.join(str(x) for x in lime_list)
    # print(lime_str)


    #lime counting features frequency 
    # for i in feat_list:
    #     if i in lime_str:
    #         #update dict
    #         feat_dict[i] = feat_dict[i] + 1
    
    c = c + 1 
    print ('progress',100*(c/samples),'%')

# Define the number you want to divide by
divider = samples

# Use a list comprehension to divide all elements by the same number
feature_val = [x / divider for x in feature_val]

# for item1, item2 in zip(feature_name, feature_val):
#     print(item1, item2)


# Use zip to combine the two lists, sort based on list1, and then unzip them
zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1],reverse=True)

# Convert the sorted result back into separate lists
sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# print(sorted_list1)
# print(sorted_list2)
print('----------------------------------------------------------------------------------------------------------------')


for item1, item2 in zip(sorted_list1, sorted_list2):
    print(item1, item2)
    with open(output_file_name, "a") as f:print(item1, item2, file = f)


for k in sorted_list1:
  with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)


with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
for k in sorted_list1:
  with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
with open(output_file_name, "a") as f:print("]", file = f)
print('---------------------------------------------------------------------------------')

# # print(feat_dict)
# # Sort values in descending order
# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print(k,v)

# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')


end = time.time()
with open(output_file_name, "a") as f:print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min',file = f)
print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating Sparsity Graph')
print('---------------------------------------------------------------------------------')
print('')
# print(feature_importance)

# feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
feature_val = sorted_list2

# col_name = 'col_name'  # Replace with the name of the column you want to extract
feature_name = sorted_list1

# Find the minimum and maximum values in the list
min_value = min(feature_val)
max_value = max(feature_val)

# Normalize the list to the range [0, 1]
normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]

# print(feature_name,normalized_list,'\n')
# for item1, item2 in zip(feature_name, normalized_list):
#     print(item1, item2)

#calculating Sparsity

# Define the threshold
threshold = 1e-10

# Initialize a count variable to keep track of values below the threshold
count_below_threshold = 0

# Iterate through the list and count values below the threshold
for value in normalized_list:
    if value < threshold:
        count_below_threshold += 1

Sparsity = count_below_threshold/len(normalized_list)
Spar = []
print('Sparsity = ',Sparsity)
X_axis = []
#----------------------------------------------------------------------------
for i in range(0, 11):
    i/10
    threshold = i/10
    for value in normalized_list:
        if value < threshold:
            count_below_threshold += 1

    Sparsity = count_below_threshold/len(normalized_list)
    Spar.append(Sparsity)
    X_axis.append(i/10)
    count_below_threshold = 0


#---------------------------------------------------------------------------

with open(output_file_name, "a") as f:print('y_axis_RF = ', Spar ,'', file = f)
with open(output_file_name, "a") as f:print('x_axis_RF = ', X_axis ,'', file = f)

plt.clf()

# Create a plot
plt.plot(X_axis, Spar, marker='o', linestyle='-')

# Set labels for the axes
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

# Set the title of the plot
plt.title('Values vs. X-Axis')

# Show the plot
# plt.show()
plt.savefig('d7_sparsity_RF_LIME.png')
plt.clf()

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
plt.savefig('d7_sparsity_DecisionTree_LIME.png')
plt.clf()
 
# Write results to output file
with open(output_file_name, "a") as f:
    print('\n--------------------------------------------------', file=f)
    print('Decision Tree', file=f)
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