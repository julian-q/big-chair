from turtle import position
from torch_geometric.data import Data, Dataset
import json
import os
from utils import load_initial
from clip import tokenize

class AnnotatedMeshDataset(Dataset):
    def __init__(self, models_path, annotations_path):
        super().__init__(None, None, None, None)
        self.models_path = models_path
        with open(annotations_path, 'r') as annotations_file:
            self.model2desc = json.load(annotations_file)
        self.max_desc_length = max([max([len(desc) for desc in descriptions]) for descriptions in self.model2desc.values()])

    def len(self):
        return len(os.listdir(self.models_path))

    def get(self, idx):
        obj_path = os.listdir(self.models_path)[idx]
        model_id = obj_path.split('.')[0]
        obj_path = os.path.join(self.models_path, obj_path)

        adj, edge_index, positions = load_initial(obj_path)
        model_descriptions = self.model2desc[model_id]
        tokenized_descriptions = tokenize(model_descriptions, context_length=self.max_desc_length)

        data = Data(x=positions, edge_index=edge_index, y=tokenized_descriptions.unsqueeze(dim=0))

        return data

