import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix 

def get_accuracy(model, dataloader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    return correct / len(dataloader.dataset)

def get_confusion_matrix(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)

    return cm

def plot_confusion_matrix(cm, class_names, save_path=None, show=False):
    '''Plot Confusion Matrix'''

    # Normalize 
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    # Build annotations
    annotations = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_norm[i, j] * 100
            annotations[i, j] = f'{count}\n{percent:.1f}%'
    
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm_norm, 
        annot = annotations,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True 
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix\n(top: count, bottom: percentage)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def get_class_metrics(cm, class_names):
    precision = cm.diagonal() / cm.sum(axis=0).clip(min=1)
    recall = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    metrics = {}

    for i, cls in enumerate(class_names):
        metrics[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i])
        }

    return metrics

def plot_training_curves(history, save_path):
    epochs = [h["epoch"] for h in history]

    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Loss Plot ----
    axes[0].plot(epochs, train_loss, label="Train")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True)

    # ---- Accuracy Plot ----
    axes[1].plot(epochs, train_acc, label="Train")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()