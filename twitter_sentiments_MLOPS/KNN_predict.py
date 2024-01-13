import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import matplotlib
matplotlib.use('Agg')

# Load embeddings and labels
embeddings_path = "../../data/processed/text_embeddings.pt"
labels_path = "../../data/processed/labels.pt"

embeddings_tensor = torch.load(embeddings_path)
labels_tensor = torch.load(labels_path)

# Convert the tensors to numpy arrays for compatibility with scikit-learn
X = embeddings_tensor.numpy()
y = np.argmax(labels_tensor.numpy(), axis=1)

# Normalize the embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets for the KNN model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predicting and evaluating the KNN model
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_report = classification_report(y_test, y_pred_knn, zero_division=0)

print("KNN Model Evaluation:")
print("Accuracy:", knn_accuracy)
print(knn_report)

# Confusion matrix with proportions
cm = confusion_matrix(y_test, y_pred_knn)
cm_proportions = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_proportions, annot=True, fmt=".2f")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for KNN Model (Proportions)')
plt.savefig('KNN_CM_Proportions.png')
