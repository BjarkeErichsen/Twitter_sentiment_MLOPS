import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import matplotlib.pyplot as plt
print('Quantization support:', 'fbgemm' in torch.backends.quantized.supported_engines)


# Define the FCNN_model class
class FCNN_model(nn.Module):
    def __init__(self):
        super(FCNN_model, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def get_embed(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model_fp32 = FCNN_model()
model_fp32.eval()

# Specify the quantization configuration
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Fuse model (if applicable, this model doesn't need fusion)

# Prepare and quantize the model
model_fp32_prepared = torch.quantization.prepare(model_fp32)
model_int8 = torch.quantization.convert(model_fp32_prepared)

# Generate 5 random tensors and run inference
outputs_fp32 = []
outputs_int8 = []

for _ in range(5):
    x = torch.randn(1, 768).to(torch.float32)
    out_fp32 = model_fp32(x)
    outputs_fp32.append(out_fp32.detach().numpy().flatten())
    out_int8 = model_int8(x)
    outputs_int8.append(out_int8.detach().numpy().flatten())

# Create subplots for each output dimension
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i in range(4):
    for j in range(5):
        # Plot for each output dimension
        axes[i].scatter(outputs_fp32[j][i], outputs_int8[j][i], label=f'Input {j+1}', alpha=0.7)
        axes[i].set_title(f'Output Dimension {i+1}')
        axes[i].set_xlabel('FP32 Output')
        axes[i].set_ylabel('INT8 Output')
        axes[i].grid(True)

plt.legend()
plt.tight_layout()
plt.show()
