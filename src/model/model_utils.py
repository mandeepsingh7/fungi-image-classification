from torchvision import models 
import torch.nn as nn 

from src.config import DROPOUT 

def build_efficientnet_model(
        experiment: str,
        num_classes: int,
        dropout: float = DROPOUT
):
    '''
    Build EfficientNetV2-S based classifier based on the `experiment type`. 
    Experiment options : 
    - `head_64` (Feature extractor -> nn.in_features -> 64 -> num_classes)
    - `head_256_64` (Feature extractor -> nn.in_features -> 256 -> 64 -> num_classes)
    - `last1_64` (Unfreeze last block -> nn.in_features -> 64 -> num_classes)
    - `last2_64` (unfreeze last two blocks -> nn.in_features -> 64 -> num_classes)

    '''
    model = models.efficientnet_v2_s(weights='DEFAULT')

    # Freeze backbone 
    for params in model.features.parameters():
        params.requires_grad = False
    
    in_features = model.classifier[1].in_features 

    if experiment in ['head_64', 'last1_64', 'last2_64']:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_classes),
        )
	
    elif experiment == 'head_256_64':
        model.classifier = nn.Sequential(
			nn.Dropout(p=dropout, inplace=True),
			nn.Linear(in_features, out_features=256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=256, out_features=64),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=64, out_features=num_classes),
        )
    else:
        raise ValueError(f"Unknown experiment type: {experiment}") 
    
    if experiment == 'last1_64':
        # Unfreeze last block
        for params in model.features[-1].parameters():
            params.requires_grad = True

    if experiment == 'last2_64':
        for params in model.features[-2:].parameters():
            params.requires_grad = True

    return model
