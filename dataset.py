from torch.utils.data import Dataset
from torch.nn.functional import pad
import json
import os
from utils import load_initial

class AnnotatedMeshDataset(Dataset):
    def __init__(self, models_path, annotations_path):
        self.models_path = models_path
        with open(annotations_path, 'r') as annotations_file:
            self.model2desc = json.load(annotations_file)
        self.max_desc_length = max([max([len(desc) for desc in descriptions]) for descriptions in self.model2desc.values()])

    def __len__(self):
        return len(os.listdir(self.models_path))

    def __getitem__(self, idx):
        obj_path = os.listdir(self.models_path)[idx]
        model_id = obj_path.split('.')[0]
        obj_path = os.path.join(self.models_path, obj_path)

        adj_info, edge_index, positions = load_initial(obj_path)
        model_descriptions = self.model2desc[model_id]

        data = {
            'verts': positions,
            'faces': adj_info['faces'],
            'adj': adj_info['adj'],
            'descs': model_descriptions
        }

        return data

    def collate(self, batch):
        max_nodes = max([item['verts'].shape[0] for item in batch])

        data = {
            'verts': [pad(item['verts'], (0, 0, 0, max_nodes - item['verts'].shape[0])) for item in batch],
            'faces': [item['faces'] for item in batch],
            'adjs': [pad(item['adj'], (0, max_nodes - item['adj'].shape[1], 0, max_nodes - item['adj'].shape[0])) for item in batch],
            'descs': [item['descs'] for item in batch]
        }

        return data

