import torch
import transformers
import datasets

print("PyTorch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)
print("Datasets Version:", datasets.__version__)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)