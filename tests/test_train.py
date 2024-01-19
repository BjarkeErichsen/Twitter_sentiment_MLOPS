import torch
from twitter_sentiments_MLOPS.train_model_sweep_wandb import LightningDataModule, LightningModel

def test_model_initialization():
    model = LightningModel(0.001)
    assert isinstance(model,LightningModel)

def test_data_initialization():
    batch_size = 32
    data_module = LightningDataModule(batch_size=batch_size)
    data_module.setup()

    # Test if train and validation datasets are created
    assert hasattr(data_module, 'train_dataset'), "Train dataset not created"
    assert hasattr(data_module, 'val_dataset'), "Validation dataset not created"

    # Check the type of the datasets
    assert isinstance(data_module.train_dataset, torch.utils.data.TensorDataset), "Train dataset is not a TensorDataset"
    assert isinstance(data_module.val_dataset, torch.utils.data.TensorDataset), "Validation dataset is not a TensorDataset"

    # Check the size of the datasets
    assert len(data_module.train_dataset) > 0, "Train dataset is empty"
    assert len(data_module.val_dataset) > 0, "Validation dataset is empty"
def test_training_batch_shape():
    batch_size = 128
    learning_rate = 0.01  # or any appropriate value

    # Initialize data module and model
    data_module = LightningDataModule(batch_size=batch_size)
    model = LightningModel(learning_rate=learning_rate)

    data_module.setup()

    # Get a batch of data
    train_loader = data_module.train_dataloader()
    x, y = next(iter(train_loader))

    # Check the shape of the inputs
    assert x.shape == torch.Size([batch_size, 768]), f"Input shape is incorrect: expected torch.Size([128, 768]), got {x.shape}"
    assert y.shape == torch.Size([batch_size, 4]), f"Output shape is incorrect: expected torch.Size([128, 4]), got {y.shape}"

def test_single_training_step():
    batch_size = 128
    learning_rate = 0.01  # Adjust as needed

    # Initialize data module and model
    data_module = LightningDataModule(batch_size=batch_size)
    model = LightningModel(learning_rate=learning_rate)

    data_module.setup()

    # Get a batch of data
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    # Simulate a training step
    loss = model.training_step(batch, batch_idx=0)

    # Check if loss is a tensor and is not NaN or Inf
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

#coverage run -m pytestcoverage report
#coverage report
