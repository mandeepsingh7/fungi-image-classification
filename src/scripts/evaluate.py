import argparse 
import torch 
import json 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.device import get_device 
from src.data.dataloader import get_transforms, get_dataloader
from src.config import TEST_DIR 
from src.model.model_utils import build_efficientnet_model 
from src.training.evaluate_utils import get_confusion_matrix, plot_confusion_matrix, get_class_metrics, plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the model to evaluate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    model_dir_path = Path(args.model_dir)
    model_path = model_dir_path / f"{model_dir_path.name}_best.pth"

    # Load Checkpoint 
    checkpoint = torch.load(model_path, map_location=device)

    experiment = checkpoint['experiment']
    num_classes = checkpoint['num_classes']
    idx_to_class = checkpoint['idx_to_class']
    class_names = list(idx_to_class.values())

    # Build model 
    model = build_efficientnet_model(experiment, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load Data 
    transforms = get_transforms()
    test_loader = get_dataloader(
        TEST_DIR, 
        transforms['test'],
        shuffle=False
    )

    # Evaluaton Directory 
    save_dir = model_dir_path / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix 
    cm = get_confusion_matrix(model, test_loader, device)
    accuracy = cm.diagonal().sum() / cm.sum()

    plot_confusion_matrix(cm, class_names, save_path=save_dir / "confusion_matrix.png")

    # Class Metrics 
    class_metrics = get_class_metrics(cm, class_names) 

    class_metrics["overall_accuracy"] = float(accuracy)

    with open(save_dir / "class_metrics.json", "w") as f:
        json.dump(class_metrics, f, indent=4)
    
    # History 
    history_path = model_dir_path / f"{model_dir_path.name}_history.json"

    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
        
        plot_training_curves(history, save_path=save_dir / "training_curves.png")
    else:
        print('No history file found.')

if __name__ == '__main__':
    main()

