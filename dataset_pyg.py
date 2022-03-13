import torch
from torch_geometric.data import Data, InMemoryDataset
import trimesh
import json
import os
import numpy as np
from tqdm import tqdm
import logging
import pickle
import spacy
from spacy.symbols import NOUN, ADJ
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR) # quiet trimesh warnings

class AnnotatedMeshDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        with open(os.path.join(root, 'annotations.json'), 'r') as annotations_file:
            self.model2desc = json.load(annotations_file)
        self.max_desc_length = max([max([len(desc) for desc in descriptions]) 
                                    for descriptions in self.model2desc.values()])
        self.parser = spacy.load("en_core_web_sm")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        os.path.join('dataset', 'annotated_models')

    # @property
    # def raw_file_names(self):
    #     return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def get_adj_noun(self, parsed_sample):
        adj_noun_str = ""
        for possible_adj in parsed_sample:
            if possible_adj.pos == ADJ:
                ancestor = possible_adj.head
                while (ancestor.dep_ != "ROOT"):
                    if ancestor.pos == NOUN:
                        break
                    ancestor = ancestor.head
                if ancestor.pos == NOUN:
                    adj_noun_str += " " + possible_adj.text + " " + ancestor.text
                else:
                    adj_noun_str += " " + possible_adj.text
        return adj_noun_str

    def process(self):
        # processing noun-adj pairs
        # adj_noun = [[self.get_adj_noun(desc) for desc in descriptions] for descriptions in self.model2desc.values()]



        # processing graphs
        graphs = []
        for obj_class in os.listdir(os.path.join('dataset', 'colored_models')):
            for obj_folder in tqdm(os.listdir(os.path.join('dataset', 'colored_models', obj_class))):
                mesh = trimesh.load(os.path.join('dataset', 'colored_models', obj_class, obj_folder, 'model.obj'), force='mesh')
                
                # model_id = obj.split('.')[0]
                full_descs = self.model2desc[obj_folder]
                adj_nouns = [self.get_adj_noun(self.parser(desc)) for desc in full_descs]
                descs = [{'full_desc': desc, 'adj_noun': adj_noun} for desc, adj_noun in zip(full_descs, adj_nouns)]
                # convert trimesh into graph, where the vertex positions are
                # the node features! we also attach an attribute that will
                # store the natural language descriptions
                unique_edges = mesh.edges_unique
                reversed_unique_edges = np.fliplr(unique_edges)
                edges = np.concatenate([unique_edges, reversed_unique_edges])

                edge_lengths = mesh.edges_unique_length
                edge_lengths = np.concatenate([edge_lengths, edge_lengths])

                # # get rid of alpha channel
                # vertex_colors = np.delete(mesh.visual.vertex_colors, 3, 1)
                if not os.path.exists(os.path.join('dataset', 'colored_models', obj_class, obj_folder, 'vertex_colors.pickle')):
                    continue
                vertex_colors_file = open(os.path.join('dataset', 'colored_models', obj_class, obj_folder, 'vertex_colors.pickle'), 'rb')
                pos2col = pickle.load(vertex_colors_file)
                rounded_pos2col = {}
                for key, value in pos2col.items():
                    rounded_key = tuple([round(c, 4) for c in key])
                    rounded_pos2col[rounded_key] = value
                vertex_colors = []
                for v in mesh.vertices:
                    vertex_colors.append([round(c, 4) for c in v])
                vertex_colors = np.array(vertex_colors)
                vertex_colors_file.close()
                node_features = np.concatenate([mesh.vertices, vertex_colors], axis=1)

                g = Data(x= torch.tensor(node_features).to(torch.float), # torch.rand(mesh.vertices.shape[0], 30),
                            edge_index=torch.tensor(edges.T),
                            edge_attr=torch.tensor(edge_lengths).to(torch.float),
                            model_id=obj_folder,
                            descs=descs)
                graphs.append(g)
        self.data, self.slices = self.collate(graphs)
        torch.save((self.data, self.slices), self.processed_paths[0])

