import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler 
from twitter_sentiments_MLOPS.models.model import SimpleNN

embedding_dim = 768
hidden_dim = 128

model = SimpleNN
inputs = torch.randn(5, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, on_trace_ready=tensorboard_trace_handler("./log/resnet34")) as prof:
    for i in range(10):
        model(embedding_dim, hidden_dim)
        prof.step()

#print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))
#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

prof.export_chrome_trace("trace_tensorboard.json")

