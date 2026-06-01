import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=256, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.out_dim = out_dim  
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),      
            nn.GELU(),                      
            nn.Dropout(0.1),             
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, precomputed_text_embs, fallback_batch_size=None):
        device = next(self.parameters()).device
        
        if precomputed_text_embs is None:
            if fallback_batch_size is None:
                raise ValueError("Bắt buộc truyền fallback_batch_size khi đầu vào Text là None")

            fallback_emb = torch.zeros((fallback_batch_size, self.out_dim), device=device)
            return fallback_emb

        emb = self.mlp(precomputed_text_embs.to(device))
        
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
            
        return emb
    
class ImageEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, out_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.variant_gate = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1)
        )

    def forward(self, list_of_img_tensors):
        device = next(self.parameters()).device
        B = len(list_of_img_tensors)
        D = self.mlp[-1].out_features 
        
        final_embs = torch.zeros((B, D), device=device)
        
        for i, imgs_tensor in enumerate(list_of_img_tensors):
            if imgs_tensor is None or imgs_tensor.shape[0] == 0:
                continue
                
            imgs_tensor = imgs_tensor.to(device)

            out = self.mlp(imgs_tensor) 
            emb_proj = F.normalize(out, dim=-1)

            gates = self.variant_gate(emb_proj) 
            weights = F.softmax(gates.squeeze(1), dim=0) 

            weighted = (weights.unsqueeze(1) * emb_proj).sum(dim=0) 

            final_embs[i] = F.normalize(weighted, dim=-1)
            
        return final_embs

class FusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),    
            nn.GELU(),                
            nn.Dropout(0.1),           

            nn.Linear(dim // 2, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(dim // 4, 1)
        )

    def forward(self, e_text, e_img, mask=None):
        g_t = self.gate_mlp(e_text)  # [B, 1]
        g_i = self.gate_mlp(e_img)   # [B, 1]
        
        gates = torch.cat([g_t, g_i], dim=1)  # [B, 2]

        if mask is not None:
            neg_inf = torch.tensor(-1e9, device=gates.device, dtype=gates.dtype)
            gates = torch.where(mask, gates, neg_inf)

        weights = F.softmax(gates, dim=1)  # [B, 2]

        fused = (weights[:, 0:1] * e_text
                 + weights[:, 1:2] * e_img)  # [B, D]
        
        fused = F.normalize(fused, dim=-1)

        return fused, weights 

class ItemTower(nn.Module):
    def __init__(self, text_in_dim=768, img_in_dim=768, embed_dim=256):
        super().__init__()

        self.text_encoder = TextEncoder(in_dim=text_in_dim, out_dim=embed_dim)
        self.img_encoder = ImageEncoder(in_dim=img_in_dim, out_dim=embed_dim)
        self.fusion_gate = FusionGate(dim=embed_dim)

    def forward(self, text_feats, img_feats):
        device = next(self.parameters()).device

        img_emb = self.img_encoder(img_feats)
        B = img_emb.shape[0] 

        if text_feats is None:
            print("text_feats is None")
            print(text_feats)
            print(img_feats)
            txt_emb = self.text_encoder(None, fallback_batch_size=B)
        else:
            txt_emb = self.text_encoder(text_feats.to(device))

        mask = torch.ones((B, 2), dtype=torch.bool, device=device)

        if text_feats is None:
            mask[:, 0] = False 
        else:
            text_sum = text_feats.abs().sum(dim=-1)
            mask[:, 0] = (text_sum > 0)
            
        img_valid = [img is not None and len(img) > 0 for img in img_feats]
        mask[:, 1] = torch.tensor(img_valid, dtype=torch.bool, device=device)

        all_missing = (mask.sum(dim=1) == 0)
        mask[all_missing, 0] = True

        final_out, gating_weights = self.fusion_gate(txt_emb, img_emb, mask=mask)

        return txt_emb, img_emb, final_out, gating_weights