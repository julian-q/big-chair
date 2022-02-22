import torch
from torch import nn
from torch import optim
from dataset import AnnotatedMeshDataset
from torch.utils.data import DataLoader
from models import CLIP, convert_weights
from clip import tokenize

BATCH_SIZE = 100
EPOCH = 32

dataset_path = '../text2mesh_preprocess/data/managable_objects/Chair/'
dataset = AnnotatedMeshDataset(dataset_path)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

model = CLIP(embed_dim=512, context_length=dataset.max_desc_length + 2, vocab_size=49408, 
             transformer_width=512, transformer_heads=8, transformer_layers=12)
  
if device == "cpu":
  model.float()
else:
  convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
  for batch in train_dataloader:
      optimizer.zero_grad()

      mesh_list, model_descriptions_list = batch # a list of lists of descriptions for each model

      meshes = torch.stack(mesh_list, dim=0).to(device)
      texts = torch.cat([tokenize(model_descriptions) for model_descriptions in model_descriptions_list], 
                        dim=0).to(device)

      logits_per_mesh, logits_per_text = model(meshes, texts)

      target_per_mesh = torch.zeros(meshes.shape[0], texts.shape[0]) # uniform distribution over matching descs
      target_per_text = torch.zeros(texts.shape[0], meshes.shape[0]) # one-hot distribution for single matching shape
      i_desc = 0
      for i_mesh, model_descriptions in model_descriptions_list:
        target_per_mesh[i_mesh, i_desc:i_desc + len(model_descriptions)] = 1 / len(model_descriptions)
        target_per_text[i_desc:i_desc + len(model_descriptions), i_mesh] = 1
        i_desc += len(model_descriptions)

      total_loss = (loss_img(logits_per_mesh, target_per_mesh) + loss_txt(logits_per_text, target_per_text)) / 2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else:
        convert_models_to_fp32(model)
        optimizer.step()
        convert_weights(model)
