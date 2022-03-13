from dataset_pyg import AnnotatedMeshDataset
import torch
import os

dataset = AnnotatedMeshDataset('dataset')
dataset.shuffle()
dataset = dataset[:5000]

train_share = int(0.8 * len(dataset))
val_share = int(0.1 * len(dataset))

train_set = dataset[:train_share]
val_set = dataset[train_share:train_share + val_share]
test_set = dataset[train_share + val_share:train_share + 2*val_share]

torch.save(train_set, os.path.join('dataset', 'processed', 'train_set.pt'))
torch.save(val_set, os.path.join('dataset', 'processed', 'val_set.pt'))
torch.save(test_set, os.path.join('dataset', 'processed', 'test_set.pt'))
