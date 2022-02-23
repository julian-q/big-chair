# based on https://github.com/openai/CLIP/issues/83
import torch
from torch import nn
from torch import optim
from dataset_pyg import AnnotatedMeshDataset
from torch.utils.data import DataLoader
from models import CLIP, SimpleMeshEncoder
from clip import tokenize

BATCH_SIZE = 5
EPOCH = 32

dataset_root = './dataset/'
# assumes that ./dataset/raw/ is full of .obj files!!!
dataset = AnnotatedMeshDataset(dataset_root)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = CLIP(joint_embed_dim=512, mesh_encoder=SimpleMeshEncoder, 
             context_length=dataset.max_desc_length + 2, vocab_size=49408, 
             transformer_width=512, transformer_heads=4, transformer_layers=6).to(device)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
    print('starting epoch', epoch)
    for i_batch, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        # now, batch contains a mega graph containing each
        # graph in the batch, as usual with pyg data loaders.
        # each of these graphs has a 'descs' array containing
        # its descriptions, all of which get combined into one
        # giant nested array. we tokenize them below:
        batch.to(device)
        batch_texts = torch.cat([tokenize(model_descs, context_length=dataset.max_desc_length + 2) 
                                for model_descs in batch.descs], 
                                dim=0).to(device)

        logits_per_mesh, logits_per_text = model(batch, batch_texts)
        # uniform distribution over matching descs
        target_per_mesh = torch.zeros(batch[0].shape[0], batch_texts.shape[0]).to(device) 
        # one-hot distribution for single matching shape
        target_per_text = torch.zeros(batch_texts.shape[0], batch[0].shape[0]).to(device) 
        i_desc = 0
        for i_mesh, model_descs in enumerate(batch['descs']):
            target_per_mesh[i_mesh, i_desc:i_desc + len(model_descs)] = 1 / len(model_descs)
            target_per_text[i_desc:i_desc + len(model_descs), i_mesh] = 1
            i_desc += len(model_descs)

        total_loss = (loss_img(logits_per_mesh, target_per_mesh) + loss_txt(logits_per_text, target_per_text)) / 2
        print('batch', i_batch, 'loss:', total_loss.item())
        total_loss.backward()
        optimizer.step()
