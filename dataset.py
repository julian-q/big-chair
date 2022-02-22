from torch.utils.data import Dataset
import json
import os
from utils import load_initial

class AnnotatedMeshDataset(Dataset):
    def __init__(self, models_path, annotations_path):
        self.models_path = models_path
        self.model2desc = json.load(annotations_path)
        self.max_desc_length = max(max(len(descriptions)) for descriptions in self.model2desc.values())

    def __len__(self):
        return len(os.listdir(self.models_path)) // 2

    def __getitem__(self, idx):
        obj_path = os.listdir(self.models_path)[idx * 2 - 1]
        model_id = obj_path.split('.')[0]

        adj_info, positions = load_initial(obj_path)
        model_descriptions = self.model2desc[model_id]

        return (adj_info, positions), model_descriptions
