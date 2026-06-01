import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricExplicitContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb, pos_item_emb, neg_item_emb):
        user_emb = F.normalize(user_emb, p=2, dim=1)
        pos_item_emb = F.normalize(pos_item_emb, p=2, dim=1)
        neg_item_emb = F.normalize(neg_item_emb, p=2, dim=1)

        sim_matrix = torch.matmul(user_emb, pos_item_emb.T) / self.temperature

        explicit_neg_logits = (user_emb * neg_item_emb).sum(dim=1, keepdim=True) / self.temperature

        logits_u2i = torch.cat([sim_matrix, explicit_neg_logits], dim=1) # [B, B+1]
        labels_u2i = torch.arange(user_emb.size(0), device=user_emb.device)
        loss_u2i = F.cross_entropy(logits_u2i, labels_u2i)

        logits_i2u = sim_matrix.T 
        labels_i2u = torch.arange(pos_item_emb.size(0), device=pos_item_emb.device)
        loss_i2u = F.cross_entropy(logits_i2u, labels_i2u)

        loss = (loss_u2i + loss_i2u) / 2.0
        return loss