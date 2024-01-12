import torch
import pytest
import os.path

file_path_labels = "data/processed/labels.pt"
file_path_embeddings = "data/processed/text_embeddings.pt"

@pytest.mark.skipif(not os.path.exists(file_path_embeddings) or not os.path.exists(file_path_labels), reason="Data files not found")
def test_data():

    embeddings = torch.load(file_path_embeddings)
    labels = torch.load(file_path_labels)

    assert all(
        embeddings[i].shape == torch.Size([768]) for i in range(len(embeddings))
    ), "embeddings data shape is wrong"

    assert all(
        labels[i].shape == torch.Size([4]) for i in range(len(labels))
    ), "labels data shape is wrong"

    vectors = [
        torch.tensor([1, 0, 0, 0]),
        torch.tensor([0, 1, 0, 0]),
        torch.tensor([0, 0, 1, 0]),
        torch.tensor([0, 0, 0, 1])
    ]
    assert all(
        [label_type in labels for label_type in vectors]
    ), "labels dont cover all categories"
    #assert all(
    #    [for label]
    #), "labels dont cover all categories"
