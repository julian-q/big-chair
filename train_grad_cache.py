import torch
from torch import nn
from torch import optim
from dataset_pyg import AnnotatedMeshDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models import MeshEncoder, DescriptionContextEncoder, HierarchicalMeshEncoder, DescriptionEncoder
from loss import ContrastiveLoss
from grad_cache import GradCache
import random
import os
from argparse import ArgumentParser
from typing import List
from evaluate_big_embeddings import evaluate
torch.autograd.set_detect_anomaly(True)
import sys

argp = ArgumentParser()
argp.add_argument('name',
    help="name of routine")
# argp.add_argument('--gnn',
# 	help='Which gnn to run ("GraphSAGE" or "GAT")',
# 	choices=['GraphSAGE', 'GAT'], default='GAT')
argp.add_argument('--adj_noun',
	help='use adj/noun pairs?', type=bool, default=False)
argp.add_argument('--epoch',
	help='number of epochs', type=int, default=100)
argp.add_argument('--batch_size',
	help='batch size', type=int, default=200)
argp.add_argument('--sub_batch_size',
	help='batch size', type=int, default=20)
argp.add_argument('--descs_per_mesh',
	help='number of descriptions per each mesh in a batch', type=int, default=5)
argp.add_argument('--joint_embedding_dim',
	help='dimension of joint embedding space', type=int, default=128)
args = argp.parse_args()

if not os.path.isdir(args.name):
	os.mkdir(args.name)
# dataset setup

train_set = torch.load(os.path.join('dataset', 'processed', 'train_set.pt'))
train_dataloader = DataLoader(train_set, batch_size=args.sub_batch_size, shuffle=False)

val_set = torch.load(os.path.join('dataset', 'processed', 'val_set.pt'))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# init models
desc_encoder = DescriptionContextEncoder(args.joint_embedding_dim, args.adj_noun).to(device)

# 6 is input dim because we have 3 for vertex positions and 3 for vertex colors
mesh_encoder = MeshEncoder(6, args.joint_embedding_dim).to(device)

contrastive_loss = ContrastiveLoss().to(device)

def split_inputs(model_input, chunk_size):
	return model_input

# gradient caching
gc = GradCache(models=[desc_encoder, mesh_encoder],
			   chunk_sizes=[args.sub_batch_size * args.descs_per_mesh,
			   				args.sub_batch_size],
			   loss_fn=contrastive_loss,
			   split_input_fn=split_inputs)

desc_encoder.train()
mesh_encoder.train()
contrastive_loss.train()
parameters = list(desc_encoder.parameters()) \
		   + list(mesh_encoder.parameters()) \
		   + list(contrastive_loss.parameters())
optimizer = optim.Adam(parameters, lr=1e-2,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 
# Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

losses = []
train_accs = []
val_accs = []

for epoch in range(args.epoch):
	print('starting epoch', epoch)

	desc_encoder.train()
	mesh_encoder.train()
	contrastive_loss.train()

	batch = []
	i_batch = 0
	for sub_batch in train_dataloader:
		sub_batch.to(device)
		batch.append(sub_batch)

		if len(batch) >= args.batch_size // args.sub_batch_size:
			optimizer.zero_grad()

			batch_descs, batch_meshes = [sub_batch.descs for sub_batch in batch], batch
			sampled_descs = [[random.choices(descs, k=args.descs_per_mesh) for descs in sub_batch_descs]
							 for sub_batch_descs in batch_descs]

			loss = gc(sampled_descs, batch_meshes) # GradCache takes care of backprop
			optimizer.step()
			loss.detach().cpu()

			print("batch " + str(i_batch) + ": " + str(loss.item()))
			average_loss = loss / (len(batch) * args.sub_batch_size)
			losses.append(average_loss)
			torch.save(losses, os.path.join(args.name, args.name + "_loss.pt"))

			#print(torch.cuda.memory_summary())


			i_batch += 1
			batch = []

	epoch_acc = evaluate(train_set[:len(val_set)], desc_encoder, mesh_encoder, args.descs_per_mesh, device="cuda:0")
	print('training accuracy:', epoch_acc)
	train_accs.append(epoch_acc)
	
	torch.save(desc_encoder.state_dict(), os.path.join(args.name, args.name + "_desc_parameters.pt"))
	torch.save(mesh_encoder.state_dict(),os.path.join(args.name, args.name + "_mesh_parameters.pt"))
	torch.save(contrastive_loss.state_dict(), os.path.join(args.name, args.name + "_loss_parameters.pt"))

	torch.save(losses, os.path.join(args.name, args.name + "_loss.pt"))
	torch.save(train_accs, os.path.join(args.name, args.name + "_train_accs.pt"))


print("done!")

print('final evaluation')
val_acc = evaluate(val_set, desc_encoder, mesh_encoder, args.descs_per_mesh, device="cuda:0")
torch.save(val_acc, os.path.join(args.name, args.name + '_val_acc.pt'))






