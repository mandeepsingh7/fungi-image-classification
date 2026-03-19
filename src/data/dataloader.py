import torch 
import numpy as np 
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader 

from src.config import BATCH_SIZE, TRAIN_DIR, TEST_DIR, VALID_DIR

def get_transforms(image_size: int = 224):
	"""Return image transformations for train/valid/test datasets."""
	img_transforms = {
		"train": transforms.Compose([
			transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),
			transforms.RandomRotation(10),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.CenterCrop(size=image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]),
		"valid": transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(size=image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]),
		"test": transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(size=image_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]),
	}
	return img_transforms

def get_dataloader(data_dir: str, img_transforms, shuffle: bool = True, batch_size: int = BATCH_SIZE):
	'''Return a DataLoader for the given data directory and transformations.'''
	dataset = datasets.ImageFolder(root=data_dir, transform=img_transforms)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
	return loader

def create_datasets_and_loaders(train_dir: str = TRAIN_DIR, valid_dir: str = VALID_DIR, test_dir: str = TEST_DIR, img_transforms=None, batch_size: int = BATCH_SIZE):
	"""Return (train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader)."""
	if img_transforms is None:
		img_transforms = get_transforms()

	train_dataset = datasets.ImageFolder(root=train_dir, transform=img_transforms["train"])
	valid_dataset = datasets.ImageFolder(root=valid_dir, transform=img_transforms["valid"])
	test_dataset = datasets.ImageFolder(root=test_dir, transform=img_transforms["test"])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader

def compute_class_weights_from_loader(dataloader, device=None):
	"""Compute class weights to handle class imbalance.
	Returns a torch tensor.
	"""
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
	num_classes = len(dataloader.dataset.classes)
	all_labels = [label for _, label in dataloader.dataset]
	counts = np.bincount(all_labels, minlength=num_classes)
	class_weights = 1.0 / (counts + 1e-6)
	class_weights = class_weights / class_weights.sum() * num_classes
	return torch.tensor(class_weights, dtype=torch.float).to(device)