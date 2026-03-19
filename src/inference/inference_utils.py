import torch 
import numpy as np 

from src.model.model_utils import build_efficientnet_model 
from src.utils.device import get_device

def load_model_from_checkpoint(model_path, device):
    '''Load a model from a checkpoint.'''
    checkpoint = torch.load(model_path, map_location=device)
    experiment = checkpoint['experiment']
    num_classes = checkpoint['num_classes']

    model = build_efficientnet_model(
        experiment, num_classes=num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    metadata = {
        'experiment': checkpoint.get('experiment'),
        'num_classes': checkpoint.get('num_classes'),
        'class_to_idx': checkpoint.get('class_to_idx'),
        'idx_to_class': checkpoint.get('idx_to_class'),
        'epoch': checkpoint.get('epoch'),
        'loss': checkpoint.get('loss')
    }

    return model, metadata

def predict_proba(model, img_tensor, device):
    '''Make a prediction using the model.'''
    model.to(device)
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)
    
    return probs.cpu()

def predict_class(model, img_tensor, device=None):
    probs = predict_proba(model, img_tensor, device)
    return probs.argmax(dim=1).item()


def get_top_k_predictions(probs, idx_to_class, top_k=3):
    top_probs, top_idx = probs.topk(top_k, dim=1)
    results = [
        {
            "class": idx_to_class[top_idx[0, i].item()],
            "probability": top_probs[0, i].item()
        }
        for i in range(top_k)
    ]
    return results

def softmax_np(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict_onnx_proba(session, img_tensor):
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    outputs = session.run(None, {"input": img_tensor.numpy()})
    probs = softmax_np(outputs[0])

    return probs

def predict_onnx(session, img_tensor, idx_to_class, top_k=3):
    probs = predict_onnx_proba(session, img_tensor)

    top_idx = np.argsort(probs, axis=1)[:, -top_k:][:, ::-1]
    top_probs = np.take_along_axis(probs, top_idx, axis=1)

    results = [
        {
            "class": idx_to_class[int(top_idx[0, i])],
            "probability": float(top_probs[0, i])
        }
        for i in range(top_k)
    ]

    return results

def format_predictions(results):
    """Format predictions"""
    lines = ["\nPredictions:"]
    lines.append("-" * 20)
    lines.append("Class → Probability")
    lines.append("-" * 20)
    for r in results:
        lines.append(f"{r['class']} → {r['probability']:.4f}")
    return "\n".join(lines)