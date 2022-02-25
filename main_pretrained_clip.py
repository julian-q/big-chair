# based on https://github.com/openai/CLIP/issues/83
import torch
from torch import nn
from torch import optim
from dataset_pyg import AnnotatedMeshDataset
from evaluate import evaluate
from torch_geometric.loader import DataLoader
from models import CLIP_pretrained, SimpleMeshEncoder
from torch.utils.tensorboard import SummaryWriter
from clip import tokenize

BATCH_SIZE = 2
EPOCH = 50


dataset_root = './dataset/'
# assumes that ./dataset/raw/ is full of .obj files!!!
dataset = AnnotatedMeshDataset(dataset_root)
dataset.shuffle()

train_share = int(len(dataset) * 0.7)
val_share = int(((len(dataset) - train_share) * 2) / 3)

train_dataset = dataset[: train_share]
val_dataset = dataset[train_share: train_share + val_share]
test_dataset = dataset[train_share + val_share: ]

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = CLIP_pretrained(joint_embed_dim=128,
                        mesh_encoder=SimpleMeshEncoder,
                        context_length=dataset.max_desc_length).to(device)
model.train()

loss_mesh = nn.CrossEntropyLoss()
loss_text = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

def eval(logits_per_text, targets_per_text):
    preds = logits_per_text.argmax(dim=1)
    labels = targets_per_text.argmax(dim=1)
    return torch.sum(preds == labels) / labels.shape[0]

writer = SummaryWriter()
total_loss = torch.tensor([0], dtype=torch.float).to(device)
grad_step = 0
count = 0
for epoch in range(EPOCH):
    print('starting epoch', epoch)
    for i_batch, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
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
        # uniform distribution over matching descs
        target_per_mesh = torch.zeros(n_batch, batch_texts.shape[0]).to(device)
        # one-hot distribution for single matching shape
        target_per_text = torch.zeros(batch_texts.shape[0], n_batch).to(device)
        # loop over the descriptions and populate above
        i_desc = 0
        for i_mesh, model_descs in enumerate(batch.descs):
            desc2mesh[i_desc:i_desc + len(model_descs)] = i_mesh
            target_per_mesh[i_mesh, i_desc:i_desc + len(model_descs)] = 1 / len(model_descs)
            target_per_text[i_desc:i_desc + len(model_descs), i_mesh] = 1
            i_desc += len(model_descs)

        logits_per_mesh, logits_per_text = model(batch, batch_texts, desc2mesh)
        acc = eval(logits_per_text, target_per_text)
        writer.add_scalar('Accu/train', acc.item(), count)
        print('train accuracy:', acc)
        total_loss += (loss_mesh(logits_per_mesh, target_per_mesh) + loss_text(logits_per_text, target_per_text)) / 2
        if i_batch % 10 == 9:
            total_loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', total_loss.item(), grad_step)
            print('batch', i_batch, 'loss:', total_loss.item())
            grad_step += 1
            total_loss = torch.tensor([0], dtype=torch.float).to(device)
            torch.save(model.state_dict(), "parameters.pt")
        count += 1

torch.save(model.state_dict(), "parameters.pt")

evaluate(val_dataset, model, device="cpu")



