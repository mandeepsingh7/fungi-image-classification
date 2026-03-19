import argparse 
import torch.nn as nn 
import torch 
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import TRAIN_DIR, VALID_DIR, TRAINING_RUNS_DIR 
from src.model.model_utils import build_efficientnet_model
from src.utils.device import get_device 
from src.data.dataloader import get_dataloader, get_transforms, compute_class_weights_from_loader
from src.training.train_utils import train
from src.utils.logging import setup_logging 

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to train")
    parser.add_argument('--experiment', type=str, default='last2_64', help="Name of the experiment")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--backbone_lr', type=float, default=1e-4, help='Learning rate for the backbone')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging(name=args.model_name)
    device = get_device()
    transforms = get_transforms()

    train_loader = get_dataloader(TRAIN_DIR, transforms['train'], shuffle = True)
    valid_loader = get_dataloader(VALID_DIR, transforms['valid'], shuffle = False)

    model = build_efficientnet_model(
        args.experiment,
        num_classes = len(train_loader.dataset.classes)
    )

    model = model.to(device)

    class_weights = compute_class_weights_from_loader(train_loader, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if name.startswith('features'):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.lr}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min') 

    save_dir = TRAINING_RUNS_DIR / args.model_name 
    save_dir.mkdir(parents=True, exist_ok=True)

    try: 
        train (
            model,
            args.model_name,
            args.experiment,
            train_loader,
            valid_loader,
            loss_fn,
            optimizer,
            scheduler,
            args.epochs,
            args.patience,
            device,
            save_dir,
            logger = logger
        )
    except Exception as e:
        logger.exception('Training crashed')
        raise 

if __name__ == "__main__":
    main()
