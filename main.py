import torch
from torch import nn
from torch import optim
from dataset import AnnotatedMeshDataset
from torch.utils.data import DataLoader
from models import CLIP, convert_weights
from clip import tokenize
from torch_geometric.utils import to_dense_adj

BATCH_SIZE = 5
EPOCH = 32

models_path = 'dataset/chair_objs/'
annotations_path = 'dataset/annotations.json'
dataset = AnnotatedMeshDataset(models_path, annotations_path)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

model = CLIP(joint_embed_dim=512, context_length=dataset.max_desc_length + 2, vocab_size=49408, 
             transformer_width=512, transformer_heads=8, transformer_layers=12)
  
if device == "cpu":
  model.float()
else:
  convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
  print('starting epoch', epoch)
  for i_batch, batch in enumerate(train_dataloader):
      optimizer.zero_grad()

      # mesh_list, model_descriptions_list = batch # a list of lists of descriptions for each model
      
      # mesh_embeds = None
      # desc_embeds = None
      # logits_per_mesh = None
      # logits_per_text = None
      # # pass through encoders
      # for mesh, adj, desc in zip(batch['verts'], batch['adjs'], batch['descs']):
      #   # mesh = mesh.cuda()
      #   # adj = adj.cuda()
      #   text = tokenize(desc, context_length=dataset.max_desc_length)
      #   if logits_per_mesh is None and logits_per_text is None: 
      #     logits_per_mesh, logits_per_text = model(mesh, text)
      #   else: 
      #     logits_mesh_i, logits_text_i = model(mesh, text)
      #     logits_per_mesh = torch.cat((logits_per_mesh, logits_mesh_i), dim=0)
      #     logits_per_text = torch.cat((logits_per_text, logits_text_i), dim=0)


      # meshes = torch.stack(mesh_list, dim=0).to(device)
      # texts = torch.cat([tokenize(model_descriptions) for model_descriptions in model_descriptions_list], 
                        # dim=0).to(device)

      # batch_positions = batch.x
      # batch_adj = to_dense_adj(batch.edge_index).squeeze(dim=0)
      # batch_mesh = (batch_positions, batch_adj)
      # texts = batch.y

      batch_adjs = torch.stack(batch['adjs'], dim=0).to(device)
      batch_positions = torch.stack(batch['verts'], dim=0).to(device)
      batch_meshes = (batch_positions, batch_adjs)
      batch_texts = torch.cat([tokenize(model_descs, context_length=dataset.max_desc_length + 2) 
                               for model_descs in batch['descs']], 
                              dim=0).to(device)

      logits_per_mesh, logits_per_text = model(batch_meshes, batch_texts)

      target_per_mesh = torch.zeros(batch_meshes[0].shape[0], batch_texts.shape[0]) # uniform distribution over matching descs
      target_per_text = torch.zeros(batch_texts.shape[0], batch_meshes[0].shape[0]) # one-hot distribution for single matching shape
      i_desc = 0
      for i_mesh, model_descs in enumerate(batch['descs']):
        target_per_mesh[i_mesh, i_desc:i_desc + len(model_descs)] = 1 / len(model_descs)
        target_per_text[i_desc:i_desc + len(model_descs), i_mesh] = 1
        i_desc += len(model_descs)

      total_loss = (loss_img(logits_per_mesh, target_per_mesh) + loss_txt(logits_per_text, target_per_text)) / 2
      print('batch', i_batch, 'loss:', total_loss.item())
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else:
        convert_models_to_fp32(model)
        optimizer.step()
        convert_weights(model)
