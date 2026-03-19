import argparse 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.inference.preprocess import get_inference_transform, load_image, preprocess_image
from src.inference.inference_utils import format_predictions, get_top_k_predictions, load_model_from_checkpoint, predict_proba
from src.utils.device import get_device


def arg_parse():
    parser = argparse.ArgumentParser(description='Run inference on a single image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to return.')
    return parser.parse_args()

def main():
    args = arg_parse()
    device = get_device()

    # Load Model 
    model, metadata = load_model_from_checkpoint(args.model_path, device)
    idx_to_class = metadata['idx_to_class']

    # Load and Preprocess image 
    image = load_image(args.image_path)
    transform = get_inference_transform()
    img_tensor = preprocess_image(image, transform)

    # Prediction
    probs = predict_proba(model, img_tensor, device)

    # Top k predictions 
    results = get_top_k_predictions(
        probs,
        idx_to_class,
        top_k = args.top_k
    )

    print(format_predictions(results))

if __name__ == '__main__':
    main()
