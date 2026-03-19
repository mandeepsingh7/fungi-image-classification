import torch
from src.config import MEAN, STD 

def denormalize(tensor, mean=MEAN, std=STD):
    if mean is None:
        mean = MEAN
    if std is None:
        std = STD
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    out = tensor.cpu() * std + mean
    out = out.clamp(0, 1)
    return out.permute(1, 2, 0).numpy()