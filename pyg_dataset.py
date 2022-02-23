import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_trimesh
import trimesh
import json
import os
from clip import tokenize
from tqdm import tqdm

class AnnotatedMeshDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        with open(os.path.join(self.root, 'annotations.json'), 'r') as annotations_file:
            self.model2desc = json.load(annotations_file)
        self.max_desc_length = max([max([len(desc) for desc in descriptions]) 
                                         for descriptions in self.model2desc.values()])
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        file_names = []
        table_objs = os.path.join(self.root, 'annotated_objs/Table')
        chair_objs = os.path.join(self.root, 'annotated_objs/Chair')
        file_names.extend([os.path.join(table_objs, file_name) 
                           for file_name in os.listdir(table_objs)])
        file_names.extend([os.path.join(chair_objs, file_name) 
                           for file_name in os.listdir(chair_objs)])
        return file_names

    @property
    def processed_file_names(self):
        file_names = []
        table_objs = os.path.join(self.root, 'annotated_objs/Table')
        chair_objs = os.path.join(self.root, 'annotated_objs/Chair')
        table_graphs = os.path.join(self.root, 'annotated_graphs/Table')
        chair_graphs = os.path.join(self.root, 'annotated_graphs/Chair')
        file_names.extend([file_name for file_name in os.listdir(table_objs)])
        file_names.extend([file_name for file_name in os.listdir(chair_objs)])
        return file_names

    def process(self):
        graphs = []
        for obj in tqdm(self.raw_file_names[:200]):
            mesh = trimesh.load(obj, force='mesh')
            descs = self.model2desc[obj.split('.')[0].split('/')[-1]]
            # convert trimesh into graph, where the vertex positions are
            # the node features! we also attach an attribute that will
            # store the natural language descriptions
            g = Data(x=torch.tensor(mesh.vertices).to(torch.float), 
                        edge_index=torch.tensor(mesh.edges.T), 
                        descs=tokenize(descs, context_length=self.max_desc_length + 2))
                        # descs=descs)
            graphs.append(g)
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])




