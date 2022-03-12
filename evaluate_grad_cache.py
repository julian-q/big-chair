# based on https://github.com/openai/CLIP/issues/83
from lib2to3.pgen2 import token
import torch
from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader
from models import DescriptionEncoder, MeshEncoder
import random
from tqdm import tqdm
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('name',
    help="name of routine")
# argp.add_argument('graph',
#     help="Which graph to run ('GraphSAGE' or 'GAT')",
#     choices=["GraphSAGE", "GAT"])
argp.add_argument('--descs_per_mesh',
	help='number of descriptions per each mesh in a batch', type=int, default=5)
args = argp.parse_args()

def batch_eval(logits_per_text, targets_per_text):
    preds = logits_per_text.argmax(dim=1)
    labels = targets_per_text.argmax(dim=1)
    return torch.sum(preds == labels) / labels.shape[0]

def top_5_eval(logits_per_text, targets_per_text, k=5):
    _, index_topk = torch.topk(logits_per_text, k=k, dim=1, sorted=False)
    target_topk = torch.gather(targets_per_text, dim=1, index=index_topk)
    return (torch.sum(torch.sum(target_topk, dim=1) > 0)) / target_topk.shape[0]

def evaluate(eval_dataset, desc_encoder, mesh_encoder, device="cpu"):
    desc_encoder.eval()
    mesh_encoder.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    desc_embeddings = torch.empty(0, 128)
    mesh_embeddings = torch.empty(0, 128)

    big_logit = torch.empty(0, len(eval_dataset))

    for batch_i in tqdm(eval_dataloader):
        print(torch.cuda.memory_summary())

        batch_i.to(device)
        batch_descs = batch_i.descs
        sampled_descs = [random.choices(descs, k=args.descs_per_mesh) for descs in batch_descs]
        tokenized_descs = desc_encoder.tokenize(sampled_descs).to(device)
            
        desc_embeddings_i = desc_encoder(tokenized_descs).cpu()

        # since each mesh may have differing numbers of descriptions, we sample a fixed
        # number (self.descs_per_mesh) of them for each mesh in order to standardize
        # memory usage

        logits_i = torch.empty(desc_embeddings.shape[0], 0)

        for batch_j in eval_dataloader:
            batch_meshes = batch_j

            mesh_embeddings = mesh_encoder(batch_meshes).cpu()

            logits_i = desc_embeddings_i @ mesh_embeddings.t()
            logits = torch.cat((logits, logits_i), dim=1)

        big_logit = torch.cat((big_logit, logits), dim=0)
        
    n_desc = desc_embeddings.shape[0]
    n_mesh = mesh_embeddings.shape[0]
    descs_per_mesh = n_desc // n_mesh

    # target distributions
    targets_per_desc = torch.zeros(n_desc, n_mesh).to(desc_embeddings.device)
    # one-hot distribution for single matching mesh
    targets_per_desc[torch.arange(n_desc), 
                        torch.arange(n_mesh).repeat_interleave(descs_per_mesh)] = 1

    total_val_acc = top_5_eval(big_logit, targets_per_desc)

    return total_val_acc.item()

device = 'cpu'
# init models
desc_encoder = DescriptionEncoder(128).to(device)
desc_encoder.load_state_dict(torch.load(args.name + "/" + args.name + "_desc_parameters.pt"))
mesh_encoder = MeshEncoder(128).to(device)
mesh_encoder.load_state_dict(torch.load(args.name + "/" + args.name + "_mesh_parameters.pt"))
val_dataset = torch.load("dataset/processed/val_set.pt")
print("Val Accuracy: ", evaluate(val_dataset, desc_encoder, mesh_encoder, device=device))

