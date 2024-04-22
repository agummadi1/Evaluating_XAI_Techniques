import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

# Set parameters
num_features = 20  # Number of features
num_labels = 6  # Number of labels
samples = 100  # Number of samples

# Load dataset
df = pd.read_csv('device7_top_20_features.csv')  # Replace 'your_dataset.csv' with your dataset filename

# Extract features and labels
X = df.drop(columns=['label'])
y = df['label']

# Min-max normalization
X_normalized = (X - X.min()) / (X.max() - X.min())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear', probability=True)  # Set probability=True
start_time = time.time()
svm.fit(X_train, y_train)
end_time = time.time()

# Calculate accuracy
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Initialize LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(),
                                 class_names=[str(i) for i in range(1, num_labels+1)], discretize_continuous=False)

# Generate Lime explanations for each sample in the test set
lime_feature_importance = np.zeros(X_train.shape[1])
hours = 0
with tqdm(total=samples, desc="Generating Lime Explanations") as pbar:
    for i in range(samples):
        exp = explainer.explain_instance(X_test.iloc[i].values, svm.predict_proba, num_features=num_features,
                                         top_labels=num_labels)
        lime_weights = exp.as_map()
        for label_idx, feature_weights in lime_weights.items():
            for (feature_idx, weight) in feature_weights:
                lime_feature_importance[feature_idx] += abs(weight)
        pbar.update(1)  # Update progress bar
        # Check progress every hour
        current_time = time.time()
        if (current_time - start_time) / 3600 >= hours + 1:
            hours += 1
            pbar.set_postfix({"Hours": hours})

# Average Lime feature importance
lime_feature_importance /= samples

# Sort Lime feature importance
sorted_indices = np.argsort(lime_feature_importance)[::-1]
sorted_features = X_train.columns[sorted_indices]
sorted_importance = lime_feature_importance[sorted_indices]

# Calculate sparsity values
sparsity_values = []
thresholds = np.linspace(0, 1, num=6)  # Adjusted to generate values in multiples of 0.2
for threshold in thresholds:
    sparsity_values.append(np.sum(sorted_importance < threshold) / len(sorted_importance))

# Write results to output file
output_file = "svm_lime_output_100.txt"
with open(output_file, 'a') as f:
    f.write("Feature Importance (Descending Order):\n")
    for feature, importance in zip(sorted_features, sorted_importance):
        f.write(f"{feature}: {importance}\n")
    f.write("\nAccuracy: {}\n".format(accuracy))
    f.write("Time taken (hours): {}\n".format((end_time - start_time) / 3600))
    f.write("Samples: {}\n".format(samples))
    f.write("Sparsity values:\n")
    for threshold, sparsity in zip(thresholds, sparsity_values):
        f.write("{:.2f}: {:.4f}\n".format(threshold, sparsity))

# # Plot sparsity graph
# plt.plot(thresholds, sparsity_values, marker='o', linestyle='-')
# plt.xlabel('Threshold')
# plt.ylabel('Sparsity')
# plt.title('Sparsity vs. Threshold')
# plt.savefig('svm_lime_sparsity.png')
# plt.show()
