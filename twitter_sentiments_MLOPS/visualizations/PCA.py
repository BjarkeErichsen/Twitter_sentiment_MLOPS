import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend, which is non-interactive and does not require a GUI
import numpy as np
import os
import torch
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    n_points_to_print = 1000
    # Load embeddings
    embeddings_path = os.path.join("data", "processed", "text_embeddings.pt")
    embeddings_tensor = torch.load(embeddings_path)

    # Load labels and convert to the true label value
    labels_path = os.path.join("data", "processed", "labels.pt")
    labels_tensor = torch.load(labels_path)
    labels = np.argmax(labels_tensor.numpy(), axis=1)

    # Convert tensor to numpy array
    embeddings = embeddings_tensor.numpy()

    # Normalize the data
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Perform PCA
    pca = PCA(n_components=4)  # Reducing to 4 dimensions
    reduced_embeddings = pca.fit_transform(normalized_embeddings)

    # Convert to a DataFrame for Seaborn
    df = pd.DataFrame(data=reduced_embeddings, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    

    # Add labels to the DataFrame
    df['Label'] = labels

    # Debugging: Check the contents of the DataFrame
    print(df.head())

    # Define a distinct color palette
    colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33F5"]  # Example colors, change as needed
    unique_labels = np.unique(labels)
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot using Seaborn's pairplot with the custom palette
    sns.pairplot(df.sample(n=n_points_to_print), hue='Label', palette=color_map)
    plt.savefig('reports/figuress/pca_pairplot.png')


    # Print the explained variance ratio
    print("Explained Variance Ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.2f}")


if __name__ == "__main__":
    main()
