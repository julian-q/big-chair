# based on https://github.com/openai/CLIP/issues/83
import torch
from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.loader import DataLoader
from models import CLIP_pretrained, SimpleMeshEncoder
import random
import os
from tqdm import tqdm

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

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    big_logit = torch.empty(len(eval_dataset), len(eval_dataset)).to(device)
    batch_i_idx = 0

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    big_logit = torch.empty(len(eval_dataset), len(eval_dataset)).to(device)
    batch_i_idx = 0

    for batch_i in tqdm(eval_dataloader):
        # print(torch.cuda.memory_summary())

        batch_i.to(device)
        batch_descs = batch_i.descs
        sampled_descs = [random.choices(descs, k=5) for descs in batch_descs]

        tokenized_descs = torch.cat([model.tokenizer(model_descs, return_tensors="pt", padding='max_length',
                                                 truncation=True).input_ids
                                 for model_descs in sampled_descs], dim=0).to(device)

        desc_embeddings_i = model.text_projection(torch.sum(model.text_encoder(tokenized_descs).last_hidden_state, dim=1)).to(device)

        # since each mesh may have differing numbers of descriptions, we sample a fixed
        # number (self.descs_per_mesh) of them for each mesh in order to standardize
        # memory usage

        logits_i = torch.empty(desc_embeddings_i.shape[0], len(eval_dataset)).to(device)
        batch_j_idx = 0

        for batch_j in eval_dataloader:
            pass
            batch_j.to(device)
            batch_meshes = batch_j

            mesh_embeddings = model.mesh_encoder(batch_meshes).to(device)

            logits_j = desc_embeddings_i @ mesh_embeddings.T
            logits_i[:, batch_j_idx:batch_j_idx + logits_j.shape[1]] = logits_j.clone()
            batch_j_idx += len(batch_j)

        big_logit[batch_i_idx:batch_i_idx + logits_i.shape[0], :] = logits_i.clone()
        batch_i_idx += logits_i.shape[0]

    n_desc = len(eval_dataloader) * args.descs_per_mesh
    n_mesh = len(eval_dataloader)
    descs_per_mesh = n_desc // n_mesh

    # target distributions
    targets_per_desc = torch.zeros(n_desc, n_mesh)
    # one-hot distribution for single matching mesh
    targets_per_desc[torch.arange(n_desc),
                     torch.arange(n_mesh).repeat_interleave(descs_per_mesh)] = 1

    total_val_acc = top_5_eval(big_logit, targets_per_desc)

    return total_val_acc.item()

    # for batch_i in tqdm(eval_dataloader):
    #     # print(torch.cuda.memory_summary())
    #
    #     batch_i.to(device)
    #     batch_descs = batch_i.descs
    #     sampled_descs = [random.choices(descs, k=5) for descs in batch_descs]
    #     sampled_tokenized = torch.cat([model.tokenizer(model_descs, return_tensors="pt", padding='max_length',
    #                                              truncation=True).input_ids
    #                              for model_descs in sampled_descs], dim=0).to(device)
    #
    #     # tokenized_descs = desc_encoder.tokenize(sampled_descs).to(device)
    #     # desc_embeddings_i = desc_encoder(tokenized_descs).to(device)
    #
    #     # since each mesh may have differing numbers of descriptions, we sample a fixed
    #     # number (self.descs_per_mesh) of them for each mesh in order to standardize
    #     # memory usage
    #
    #     emb_size = model.joint_embed_dim * 5
    #
    #     logits_i = torch.empty(sampled_tokenized.shape[0], len(eval_dataset)).to(device)
    #     batch_j_idx = 0
    #
    #     for batch_j in eval_dataloader:
    #         pass
    #         batch_j.to(device)
    #         batch_meshes = batch_j
    #
    #         # mesh_embeddings = mesh_encoder(batch_meshes).to(device)
    #
    #         _, logits_j = model(batch_meshes, sampled_tokenized)
    #
    #         logits_i[:, batch_j_idx:batch_j_idx + logits_j.shape[1]] = logits_j.clone()
    #         batch_j_idx += len(batch_j)
    #
    #         print("hi")
    #
    #     big_logit[batch_i_idx:batch_i_idx + logits_i.shape[0], :] = logits_i.clone()
    #     batch_i_idx += logits_i.shape[0]
    #
    # n_desc = len(eval_dataloader) * args.descs_per_mesh
    # n_mesh = len(eval_dataloader)
    # descs_per_mesh = n_desc // n_mesh
    #
    # # target distributions
    # targets_per_desc = torch.zeros(n_desc, n_mesh)
    # # one-hot distribution for single matching mesh
    # targets_per_desc[torch.arange(n_desc),
    #                  torch.arange(n_mesh).repeat_interleave(descs_per_mesh)] = 1
    #
    # total_val_acc = top_5_eval(big_logit, targets_per_desc)
    # return total_val_acc.item()





    # total_val_acc = torch.tensor([0], dtype=torch.float).to(device)
    # for i_batch, batch in enumerate(eval_dataloader):
    #
    #     n_batch = batch.batch.max() + 1
    #     # now, batch contains a mega graph containing each
    #     # graph in the batch, as usual with pyg data loaders.
    #     # each of these graphs has a 'descs' array containing
    #     # its descriptions, all of which get combined into one
    #     # giant nested array. we tokenize them below:
    #     batch.to(device)
    #     # we could honestly move this code into the model's forward
    #     # function now that we're using pyg
    #     batch_texts = torch.cat([model.tokenizer(model_descs, return_tensors="pt", padding='max_length',
    #                                              truncation=True).input_ids
    #                              for model_descs in batch.descs], dim=0).to(device)
    #     # vector mapping each description to its mesh index
    #     desc2mesh = torch.zeros(batch_texts.shape[0], dtype=torch.long)
    #     # one-hot distribution for single matching shape
    #     target_per_text = torch.zeros(batch_texts.shape[0], n_batch).to(device)
    #     # loop over the descriptions and populate above
    #     i_desc = 0
    #     for i_mesh, model_descs in enumerate(batch.descs):
    #         desc2mesh[i_desc:i_desc + len(model_descs)] = i_mesh
    #         target_per_text[i_desc:i_desc + len(model_descs), i_mesh] = 1
    #         i_desc += len(model_descs)
    #
    #     logits_per_mesh, logits_per_text = model(batch, batch_texts, desc2mesh)
    #     total_val_acc += top_5_eval(logits_per_text, target_per_text)
    #     del desc2mesh
    #     del target_per_text
    # return total_val_acc.item()

# dataset_root = './dataset/'
# dataset = AnnotatedMeshDataset(dataset_root)
model = CLIP_pretrained(joint_embed_dim=128,
                        mesh_encoder=SimpleMeshEncoder,
                        context_length=77,
                        opt=args.graph).to("cpu")
val_dataset = torch.load("dataset/processed/val_set.pt")
print("Val Accuracy: ", evaluate(val_dataset, model, os.path.join(args.name, args.name + "_parameters.pt")))

