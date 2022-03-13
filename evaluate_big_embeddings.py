# based on https://github.com/openai/CLIP/issues/83
from lib2to3.pgen2 import token
import torch
from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader
from models import DescriptionContextEncoder, MeshEncoder, CLIP_pretrained, SimpleMeshEncoder
import random
from tqdm import tqdm
import argparse
import os
import gc

def batch_eval(logits_per_text, targets_per_text):
    preds = logits_per_text.argmax(dim=1)
    labels = targets_per_text.argmax(dim=1)
    return torch.sum(preds == labels) / labels.shape[0]

def top_5_eval(logits_per_text, targets_per_text, k=5):
    _, index_topk = torch.topk(logits_per_text, k=k, dim=1, sorted=False)
    target_topk = torch.gather(targets_per_text, dim=1, index=index_topk)
    return (torch.sum(torch.sum(target_topk, dim=1) > 0)) / target_topk.shape[0]

def evaluate(eval_dataset, desc_encoder, mesh_encoder, descs_per_mesh, batch_size=1, device="cpu"):
    desc_encoder.eval()
    mesh_encoder.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    desc_embeddings = torch.empty(len(eval_dataset) * descs_per_mesh, 128).to(device)
    mesh_embeddings = torch.empty(len(eval_dataset), 128).to(device)
    desc_index = 0
    mesh_index = 0

    for batch in tqdm(eval_dataloader):
        batch.to(device)
        batch_descs = batch.descs
        sampled_descs = [random.choices(descs, k=args.descs_per_mesh) for descs in batch_descs]
            
        desc_embeddings_i = desc_encoder(sampled_descs).to(device)
        desc_embeddings[desc_index:desc_index + desc_embeddings_i.shape[0], :] = desc_embeddings_i
        desc_index += desc_embeddings_i.shape[0]

        mesh_embeddings_i = mesh_encoder(batch).to(device)
        mesh_embeddings[mesh_index:mesh_index + mesh_embeddings_i.shape[0], :] = mesh_embeddings_i
        mesh_index += mesh_embeddings_i.shape[0]

    big_logits = desc_embeddings @ mesh_embeddings.T
        
    n_desc = len(eval_dataloader) * descs_per_mesh
    n_mesh = len(eval_dataloader)

    # target distributions
    targets_per_desc = torch.zeros(n_desc, n_mesh)
    # one-hot distribution for single matching mesh
    targets_per_desc[torch.arange(n_desc), 
                     torch.arange(n_mesh).repeat_interleave(descs_per_mesh)] = 1

    total_val_acc = top_5_eval(big_logits, targets_per_desc)

    return total_val_acc.item()

if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument('name',
        help="name of routine")
    # argp.add_argument('graph',
    #     help="Which graph to run ('GraphSAGE' or 'GAT')",
    #     choices=["GraphSAGE", "GAT"])
    argp.add_argument('--descs_per_mesh',
        help='number of descriptions per each mesh in a batch', type=int, default=5)
    args = argp.parse_args()

    device = 'cpu'
    # init models

    desc_encoder = DescriptionContextEncoder(128).to(device)
    desc_encoder.load_state_dict(torch.load(args.name + "/" + args.name + "_desc_parameters.pt"))
    mesh_encoder = MeshEncoder(128).to(device)
    mesh_encoder.load_state_dict(torch.load(args.name + "/" + args.name + "_mesh_parameters.pt"))
    val_dataset = torch.load("dataset/processed/val_set.pt")
    print("Val Accuracy: ", evaluate(val_dataset, desc_encoder, mesh_encoder, device=device))


