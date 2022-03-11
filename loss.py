from sqlalchemy import desc
import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.desc_loss_fn = nn.CrossEntropyLoss()
        self.mesh_loss_fn = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, desc_embeddings, mesh_embeddings):
        n_desc = desc_embeddings.shape[0]
        n_mesh = mesh_embeddings.shape[0]
        descs_per_mesh = n_desc // n_mesh

        # predicted logits
        logits_per_mesh = self.logit_scale.exp() * mesh_embeddings @ desc_embeddings.T
        logits_per_desc = logits_per_mesh.T

        # target distributions
        targets_per_desc = torch.zeros(n_desc, n_mesh).to(desc_embeddings.device)
        # one-hot distribution for single matching mesh
        targets_per_desc[torch.arange(n_desc), 
                         torch.arange(n_mesh).repeat_interleave(descs_per_mesh)] = 1
        targets_per_mesh = torch.zeros(n_mesh, n_desc).to(mesh_embeddings.device)
        # uniform distribution over matching descs
        targets_per_mesh[torch.arange(n_mesh).unsqueeze(dim=1), 
                         torch.arange(n_desc).reshape(n_desc // descs_per_mesh, descs_per_mesh)] = 1 / descs_per_mesh

        desc_loss = self.desc_loss_fn(logits_per_desc, targets_per_desc)
        mesh_loss = self.desc_loss_fn(logits_per_mesh, targets_per_mesh)
        total_loss = (desc_loss + mesh_loss) / 2
        return total_loss
