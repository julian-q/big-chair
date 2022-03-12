import torch
import matplotlib.pyplot as plt

losses = torch.load('gat_grad_cache/gat_grad_cache_loss.pt')

for i, loss in enumerate(losses):
    losses[i] = loss.cpu()

plt.plot(losses)
plt.show()