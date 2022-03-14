from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader
from models import DescriptionContextEncoder, MeshEncoder
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import os

dataset = torch.load('dataset/processed/val_set.pt')
retrieval_dataset = dataset[:20]
retrieval_dataloader = DataLoader(retrieval_dataset, batch_size=2, shuffle=False)

desc_encoder = DescriptionContextEncoder(128, adj_noun=True)
mesh_encoder = MeshEncoder(6, 128)
desc_encoder.load_state_dict(torch.load('simple_context/simple_context_desc_parameters.pt', map_location=torch.device('cpu')))
mesh_encoder.load_state_dict(torch.load('simple_context/simple_context_mesh_parameters.pt', map_location=torch.device('cpu')))

query_index = random.randint(0, len(retrieval_dataset) - 1)
query_desc = [random.sample(retrieval_dataset[query_index].descs, 1)]
print('query:', query_desc[0][0]['full_desc'])

query_desc_embedding = desc_encoder(query_desc)
print('embed:', query_desc_embedding)

mesh_embeddings = torch.empty(len(retrieval_dataset), 128)
mesh_index = 0

for batch in tqdm(retrieval_dataloader):
    batch_mesh_embeddings = mesh_encoder(batch)
    mesh_embeddings[mesh_index:mesh_index + batch_mesh_embeddings.shape[0]] = batch_mesh_embeddings
    mesh_index += batch_mesh_embeddings.shape[0]

logits = (query_desc_embedding @ mesh_embeddings.T).squeeze()
probabilities = F.softmax(logits, dim=0)

k = 5
_, topk_indices = torch.topk(probabilities, k=k)
topk_meshes = retrieval_dataset[topk_indices]
topk_model_ids = [m.model_id for m in topk_meshes]

print('top 5 closest models:')
for i in range(k):
    print(topk_model_ids[i], ':', probabilities[i].item())
print('to get images, enter model id into ShapeNet here: https://shapenet.org/model-querier')

