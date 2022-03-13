# based on https://github.com/openai/CLIP/issues/83
import torch
from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader
from models import CLIP_pretrained, SimpleMeshEncoder
import os

import argparse

argp = argparse.ArgumentParser()
argp.add_argument('name',
    help="name of routine")
argp.add_argument('graph',
    help="Which graph to run ('GraphSAGE' or 'GAT')",
    choices=["GraphSAGE", "GAT"])
args = argp.parse_args()

def batch_eval(logits_per_text, targets_per_text):
    preds = logits_per_text.argmax(dim=1)
    labels = targets_per_text.argmax(dim=1)
    return torch.sum(preds == labels) / labels.shape[0]

def top_5_eval(logits_per_text, targets_per_text, k=5):
    _, index_topk = torch.topk(logits_per_text, k=k, dim=1, sorted=False)
    target_topk = torch.gather(targets_per_text, dim=1, index=index_topk)
    return (torch.sum(torch.sum(target_topk, dim=1) > 0)) / target_topk.shape[0]

def evaluate(eval_dataset, model, parameters_path, device="cpu"):
    if parameters_path != None:
        model.load_state_dict(torch.load(parameters_path, map_location=torch.device(device)))
    model.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    total_val_acc = torch.tensor([0], dtype=torch.float).to(device)
    for i_batch, batch in enumerate(eval_dataloader):
        n_batch = batch.batch.max() + 1
        # now, batch contains a mega graph containing each
        # graph in the batch, as usual with pyg data loaders.
        # each of these graphs has a 'descs' array containing
        # its descriptions, all of which get combined into one
        # giant nested array. we tokenize them below:
        batch.to(device)
        # we could honestly move this code into the model's forward
        # function now that we're using pyg
        batch_texts = torch.cat([model.tokenizer(model_descs, return_tensors="pt", padding='max_length',
                                                 truncation=True).input_ids
                                 for model_descs in batch.descs], dim=0).to(device)
        # vector mapping each description to its mesh index
        desc2mesh = torch.zeros(batch_texts.shape[0], dtype=torch.long)
        # one-hot distribution for single matching shape
        target_per_text = torch.zeros(batch_texts.shape[0], n_batch).to(device)
        # loop over the descriptions and populate above
        i_desc = 0
        for i_mesh, model_descs in enumerate(batch.descs):
            desc2mesh[i_desc:i_desc + len(model_descs)] = i_mesh.clone()
            target_per_text[i_desc:i_desc + len(model_descs), i_mesh] = 1
            i_desc += len(model_descs)

        logits_per_mesh, logits_per_text = model(batch, batch_texts, desc2mesh)
        total_val_acc += top_5_eval(logits_per_text, target_per_text)
    return total_val_acc.item()

# dataset_root = './dataset/'
# dataset = AnnotatedMeshDataset(dataset_root)
model = CLIP_pretrained(joint_embed_dim=128,
                        mesh_encoder=SimpleMeshEncoder,
                        context_length=77,
                        opt=args.graph).to("cpu")
val_dataset = torch.load("dataset/processed/val_set.pt")
print("Val Accuracy: ", evaluate(val_dataset, model, os.path.join(args.name, args.name + "_parameters.pt")))

