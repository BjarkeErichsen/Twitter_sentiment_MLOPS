import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load embeddings and labels
embeddings_path = "data/processed/text_embeddings.pt"
labels_path = "data/processed/labels.pt"
embeddings_tensor = torch.load(embeddings_path)
labels_tensor = torch.load(labels_path)
X = embeddings_tensor.numpy()
y = np.argmax(labels_tensor.numpy(), axis=1)

# Normalize the embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save the model
model_filename = 'models/knn_model.joblib'
joblib.dump(knn, model_filename)