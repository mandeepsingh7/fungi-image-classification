# Fungi Image Classification

In this project, we classify fungi images into five distinct types using pretrained deep learning models: EfficientNet V2 and ResNet152. The fungi types are:

- H1 (TSH)
- H2 (BASH)
- H3 (GMA)
- H5 (SHC)
- H6 (BBH)

## Dataset

Our dataset contains images categorized into five folders, each representing a class.

## Data Preparation

1. **Data Splitting**:
   - The dataset is divided into train, test, and validation sets.
   - Three main folders (`train`, `test`, `valid`) are created, each containing five subfolders corresponding to each class.
   - Images are populated in these folders based on their respective sets and classes.

2. **Image Transformations**:
   - Different image transformations are specified for the train, test, and validation sets to augment the data and improve model robustness.

3. **Data Loaders**:
   - Train, validation, and test datasets are created based on the specified transformations.
   - DataLoaders are created for these datasets with a batch size of 16 for efficient processing.

## Models

### EfficientNet V2

- **Pretrained Model**: EfficientNet V2 Medium
- **Modified Classifier**:
  ```python
  Sequential (
    (0): Dropout(p=0.3, inplace=True)
    (1): Linear(in_features=1280, out_features=512, bias=True)
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): Linear(in_features=512, out_features=5, bias=True)
  )
- We freeze all other weights except the ones corresponding to these last 3 layers.
- Trainable Parameters: 921,093
- Epochs: 50
- Patience: 10 (Early stopping at epoch 29)
- Accuracy: 65.64%
  
### ResNet152

- **Pretrained Model**: ResNet152
- **Modified Classifier**:
  ```python
  Sequential (
  (0): Linear(in_features=2048, out_features=512, bias=True)
  (1): Linear(in_features=512, out_features=512, bias=True)
  (2): Linear(in_features=512, out_features=64, bias=True)
  (3): Linear(in_features=64, out_features=4, bias=True)
  )
- We freeze all other weights except the ones corresponding to these last 3 layers.
- Trainable Parameters: 1,344,901
- Epochs: 50
- Patience: 10 (Early stopping at epoch 44)
- Accuracy: 70.76%

## Results
- EfficientNet V2: Achieved an accuracy of 65.64% with early stopping at epoch 29.
- ResNet152: Achieved an accuracy of 70.76% with early stopping at epoch 44.

## Conclusion
This project demonstrates the fine-tuning of pretrained deep learning models for the task of fungi image classification. ResNet152 outperformed EfficientNet V2 in terms of accuracy, highlighting its effectiveness for this particular dataset.

