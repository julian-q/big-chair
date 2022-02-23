from pyg_dataset import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

dataset_root = './dataset/'
dataset = AnnotatedMeshDataset(dataset_root)
train_dataloader = DataLoader(dataset)

for batch in train_dataloader:
    print(batch.x.shape)


