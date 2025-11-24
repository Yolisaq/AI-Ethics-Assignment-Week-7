# AI Fairness Audit: COMPAS Raw Dataset (Fixed Age Parsing)

# Libraries
import pandas as pd
import numpy as np
from datetime import datetime
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load CSV
df = pd.read_csv("compas-scores-raw.csv")
df.columns = df.columns.str.strip()

# 2️⃣ Correctly parse DateOfBirth
# Change format according to your CSV, example: '%Y-%m-%d'
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
today = pd.to_datetime('today')
df['age'] = (today - df['DateOfBirth']).dt.days // 365

# Remove invalid ages
df = df[df['age'] > 0]

# Remove rows with missing critical values
df = df.dropna(subset=['age', 'Sex_Code_Text', 'Ethnic_Code_Text', 'RawScore'])

# Map Sex and Race
df['sex'] = df['Sex_Code_Text'].map({'Male': 1, 'Female': 0})
df['race'] = df['Ethnic_Code_Text'].apply(lambda x: 1 if x.upper() in ['CAUCASIAN','WHITE'] else 0)

# 3️⃣ Target label with threshold adjustment
threshold = 5  # adjust if needed
df['target'] = df['RawScore'].apply(lambda x: 1 if x >= threshold else 0)

# Check class distribution
print("Class distribution:\n", df['target'].value_counts())

# 4️⃣ Features and label
features = ['age', 'sex', 'race']
label = 'target'
df_model = df[features + [label]]

# Optional: Undersample majority class if still imbalanced
count_class_0 = df_model[df_model['target']==0].shape[0]
count_class_1 = df_model[df_model['target']==1].shape[0]
if count_class_0 > 1.5 * count_class_1:
    df_class_0 = df_model[df_model['target']==0].sample(count_class_1*2, random_state=42)
    df_class_1 = df_model[df_model['target']==1]
    df_model = pd.concat([df_class_0, df_class_1])
    print("Balanced dataset shape:", df_model.shape)

# 5️⃣ Convert to AIF360 BinaryLabelDataset
dataset = BinaryLabelDataset(df=df_model,
                             label_names=[label],
                             protected_attribute_names=['race'],
                             favorable_label=0,
                             unfavorable_label=1)

# 6️⃣ Fairness audit before mitigation
metric_before = BinaryLabelDatasetMetric(dataset,
                                         privileged_groups=[{'race': 1}],
                                         unprivileged_groups=[{'race': 0}])
print("Disparate Impact (Before Mitigation):", metric_before.disparate_impact())

# 7️⃣ Bias mitigation: Reweighing
RW = Reweighing(unprivileged_groups=[{'race': 0}],
                privileged_groups=[{'race': 1}])
dataset_transf = RW.fit_transform(dataset)

# 8️⃣ Train Logistic Regression
X_train = dataset_transf.features
y_train = dataset_transf.labels.ravel()
weights = dataset_transf.instance_weights

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train, sample_weight=weights)

# 9️⃣ Predictions
y_pred = model.predict(X_train)
dataset_pred = dataset_transf.copy()
dataset_pred.labels = y_pred

# 10️⃣ Fairness after mitigation
metric_post = ClassificationMetric(dataset_transf, dataset_pred,
                                   unprivileged_groups=[{'race': 0}],
                                   privileged_groups=[{'race': 1}])
print("Disparate Impact (After Mitigation):", metric_post.disparate_impact())
print("Equal Opportunity Difference:", metric_post.equal_opportunity_difference())

# 11️⃣ Model performance
accuracy = accuracy_score(y_train, y_pred)
print("Model Accuracy:", round(accuracy,4))

cm = confusion_matrix(y_train, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk','High Risk'], yticklabels=['Low Risk','High Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 12️⃣ Fairness improvement visualization
metrics = [metric_before.disparate_impact(), metric_post.disparate_impact()]
labels_plot = ['Before Mitigation','After Mitigation']

plt.bar(labels_plot, metrics, color=['red','green'])
plt.ylabel('Disparate Impact')
plt.title('COMPAS Fairness Audit')
plt.show()

# 13️⃣ Side-by-side comparison of actual vs predicted
comparison = df_model.copy()
comparison['Predicted'] = y_pred
print(comparison.head(10))
