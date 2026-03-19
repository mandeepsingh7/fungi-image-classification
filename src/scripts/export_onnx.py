import argparse
import torch
import json 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.model.model_utils import build_efficientnet_model
from src.inference.inference_utils import load_model_from_checkpoint
from src.utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    model_path = Path(args.model_path)

    # Load model 
    model, metadata = load_model_from_checkpoint(
        str(model_path),
        device
    )

    model = model.to("cpu")
    model.eval()

    if args.output_path:
        onnx_path = Path(args.output_path)
    else:
        onnx_path = model_path.with_suffix(".onnx")

    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to('cpu')

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=21
    )

    print(f"\nONNX model saved to: {onnx_path}")

    # Save metadata
    metadata_path = onnx_path.with_name(onnx_path.stem + "_metadata_onnx.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()