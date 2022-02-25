import torch
from torch_geometric.data import Data, InMemoryDataset
import trimesh
import json
import os
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
        for obj in tqdm(self.raw_file_names[:10]):
            mesh = trimesh.load(os.path.join(self.raw_dir, obj), force='mesh')
            model_id = obj.split('.')[0]
            descs = self.model2desc[model_id]
            # convert trimesh into graph, where the vertex positions are
            # the node features! we also attach an attribute that will
            # store the natural language descriptions
            g = Data(x=torch.rand(mesh.vertices.shape[0], 30), # torch.tensor(mesh.vertices).to(torch.float), 
                        edge_index=torch.tensor(mesh.edges.T),
                        model_id=model_id,
                        descs=descs)
            graphs.append(g)
        self.data, self.slices = self.collate(graphs)
        torch.save((self.data, self.slices), self.processed_paths[0])




