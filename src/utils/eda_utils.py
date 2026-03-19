from collections import Counter 
import seaborn as sns 
import matplotlib.pyplot as plt 
import torch
import numpy as np 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.image_utils import denormalize 
from src.config import MEAN, STD 

def get_class_count(dataloader):
    counter = Counter(dataloader.dataset.targets)
    idx_to_class = {v: k for k, v in dataloader.dataset.class_to_idx.items()}
    class_counts = {idx_to_class[idx]: count for idx, count in counter.items()}
    return class_counts

def plot_class_distribution(train_dataloader, valid_dataloader):
    train_class_counts = get_class_count(train_dataloader)
    valid_class_counts = get_class_count(valid_dataloader)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    class_counts_list = [
        ('Train', train_class_counts, axes[0]),
        ('Valid', valid_class_counts, axes[1])
    ]

    for idx, (title, class_counts, ax) in enumerate(class_counts_list):
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        total = sum(counts)
        percentages = [f'{(count / total) * 100:.1f}%' for count in counts]

        sns.barplot(x=classes, y=counts, ax=ax)
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax.text(i, count + max(counts)*0.01, pct, ha='center', va='bottom', fontsize=9)

        ax.set_title(f'{title} Set', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fungi Class')
        if idx == 0:
            ax.set_ylabel('Number of Samples')
        else:
            ax.set_ylabel('')

    plt.suptitle('Class Distribution Across Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def display_sample_images(train_dataloader, samples_per_class=10, save_path=None, mean=MEAN, std=STD):
    fig_w = 10
    fig_h = 5

    train_dataset = train_dataloader.dataset
    class_to_idx = train_dataset.class_to_idx 
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    class_images = {i: [] for i in range(num_classes)}

    for images, labels in train_dataloader:
        for img, label in zip(images, labels):
            label = label.item()
            if len(class_images[label]) < samples_per_class:
                class_images[label].append(img)
        if all(len(v) == samples_per_class for v in class_images.values()):
            break

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(fig_w, fig_h))
    for row in range(num_classes):
        for col in range(samples_per_class):
            img = class_images[row][col]
            img = denormalize(img, mean, std)
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')
        axes[row, 0].text(-0.1, 0.5, idx_to_class[row],
                        transform=axes[row, 0].transAxes,
                        fontsize=10, va='center', ha='right')

    fig.suptitle('Sample Images')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()