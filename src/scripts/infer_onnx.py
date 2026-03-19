import argparse
import json
import onnxruntime as ort
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.inference.preprocess import (
    load_image,
    preprocess_image,
    get_inference_transform
)
from src.inference.inference_utils import (
    predict_onnx,
    format_predictions
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX inference")

    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load ONNX session
    session = ort.InferenceSession(
        args.onnx_path,
        providers=["CPUExecutionProvider"]
    )

    # Load metadata
    with open(args.metadata_path) as f:
        metadata = json.load(f)

    idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}

    # Load image
    image = load_image(args.image_path)
    transform = get_inference_transform()
    img_tensor = preprocess_image(image, transform)

    # Predict
    results = predict_onnx(
        session,
        img_tensor,
        idx_to_class,
        top_k=args.top_k
    )

    # Output
    print(format_predictions(results))


if __name__ == "__main__":
    main()