# Fungi Image Classification
- In this project, we are trying to classify fungi images into 5 fungi types:
  - H1 (TSH)
  - H2 (BASH)
  - H3 (GMA)
  - H5 (SHC)
  - H6  (BBH)
- Our raw data contains 5 folders, each folder containing images belonging to a particular class.
- We first divide the data into train, test and valid.
- Then we create three folders train, test, and valid. In each folder, we create 5 folders corresponding to each class. We populate the images in these folders based on whether they belong to train, test, valid set and based on their class.
- We specify different image transforms for train, test and valid set.
- We create train, valid and test datasets based on these transformations.
- Then we use these datasets to create DataLoaders with batch size = 16
- Then we use 2 pretrained models, efficientnet_v2_m and resnet152 and finetune them on fungi images dataset. 
- In efficientnet_v2_m, we replace the classifier layer\
    Sequential (\
  (0): Dropout(p=0.3, inplace=True)\
  (1): Linear(in_features=1280, out_features=1000, bias=True)\
)\
with \
Sequential (\
  (0): Dropout(p=0.3, inplace=True)\
  (1): Linear(in_features=1280, out_features=512, bias=True)\
  (2): Linear(in_features=512, out_features=512, bias=True)\
  (3): Linear(in_features=512, out_features=5, bias=True)\
)
- We freeze all other weights except the ones corresponding to these last 3 layers.
- Trainable params: 921,093
- With efficientnet_v2_m, we get an accuracy of 65.64 %.
- We specified number of epochs = 50 and patience = 10.
- But we observed early stop at Epoch Number 29
- -----------------
- In resnet152, we replace fc layer\
  Linear(in_features=2048, out_features=1000, bias=True)\
  with\
  Sequential (\
  (0): Linear(in_features=2048, out_features=512, bias=True)\
  (1): Linear(in_features=512, out_features=512, bias=True)\
  (2): Linear(in_features=512, out_features=64, bias=True)\
  (3): Linear(in_features=64, out_features=4, bias=True)\
)
- We freeze all other weights except the ones corresponding to these last 3 layers.
- Trainable params: 1,344,901
- With resnet152, we get an accuracy of 70.76 %.
- We specified number of epochs = 50 and patience = 10.
- But we observed early stop at Epoch Number 44
