import torch
from torch_geometric.data import Data, InMemoryDataset
import trimesh
import json
import os
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR) # quiet trimesh warnings

class AnnotatedMeshDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        with open(os.path.join(root, 'annotations.json'), 'r') as annotations_file:
            self.model2desc = json.load(annotations_file)
        self.max_desc_length = max([max([len(desc) for desc in descriptions]) 
                                    for descriptions in self.model2desc.values()])
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        graphs = []
        for obj in tqdm(self.raw_file_names):
            mesh = trimesh.load(os.path.join(self.raw_dir, obj), force='mesh')
            model_id = obj.split('.')[0]
            descs = self.model2desc[model_id]
            # convert trimesh into graph, where the vertex positions are
            # the node features! we also attach an attribute that will
            # store the natural language descriptions

            unique_edges = mesh.edges_unique
            reversed_unique_edges = np.fliplr(unique_edges)
            edges = np.concatenate([unique_edges, reversed_unique_edges])

            edge_lengths = mesh.edges_unique_length
            edge_lengths = np.concatenate([edge_lengths, edge_lengths])

            g = Data(x= torch.tensor(mesh.vertices).to(torch.float), # torch.rand(mesh.vertices.shape[0], 30),
                        edge_index=torch.tensor(edges.T),
                        edge_attr=torch.tensor(edge_lengths).to(torch.float),
                        model_id=model_id,
                        descs=descs)
            graphs.append(g)
        self.data, self.slices = self.collate(graphs)
        torch.save((self.data, self.slices), self.processed_paths[0])




