from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader

dataset_root = './dataset/'
# assumes that ./dataset/raw/ is full of .obj files
dataset = AnnotatedMeshDataset(dataset_root)
train_dataloader = DataLoader(dataset, batch_size=5)

for batch in train_dataloader:
    print(batch.descs)


