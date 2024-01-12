import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is non-interactive and does not require a GUI

import os
import torch
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # Load embeddings
    # Path to the embeddings file
    embeddings_path = os.path.join("..", "..", "data", "processed", "text_embeddings.pt")
    embeddings_tensor = torch.load(embeddings_path)

    # Convert tensor to numpy array if it's not already
    embeddings = embeddings_tensor.numpy()

    # Normalize the data
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Perform PCA
    pca = PCA(n_components=4)  # Reducing to 4 dimensions
    reduced_embeddings = pca.fit_transform(normalized_embeddings)

    # Convert to a DataFrame for Seaborn
    df = pd.DataFrame(data=reduced_embeddings, columns=['PC1', 'PC2', 'PC3', 'PC4'])

    # Plot using Seaborn's pairplot
    sns.pairplot(df)
    plt.savefig('pca_pairplot.png')

    # Print the explained variance ratio
    print("Explained Variance Ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.2f}")

if __name__ == "__main__":
    main()
