import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v1, v2):
        device = v1.device
        sim_matrix = torch.matmul(v1, v2.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0), dtype=torch.long, device=device)

        loss_v1_to_v2 = F.cross_entropy(sim_matrix, labels)
        loss_v2_to_v1 = F.cross_entropy(sim_matrix.T, labels)

        loss = (loss_v1_to_v2 + loss_v2_to_v1) / 2.0

        return loss