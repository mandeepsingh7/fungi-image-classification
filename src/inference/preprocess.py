from torchvision import transforms 
from PIL import Image 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image(image_path: str):
    return Image.open(image_path).convert('RGB')

def preprocess_image(image, transform):
    return transform(image)