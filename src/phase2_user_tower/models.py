import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x: Shape [seq_len, batch_size, d_model] """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GatingFusionLayer(nn.Module):
    def __init__(self, item_emb_dim=256, text_emb_dim=256, d_model=256):
        super().__init__()
        hidden_dim = item_emb_dim*2

        self.proj_item = nn.Sequential(
            nn.Linear(item_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),   
            nn.Linear(hidden_dim, d_model),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(item_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),   
            nn.Linear(hidden_dim, d_model),
        )
        
        gate_input_dim = item_emb_dim + text_emb_dim
        self.gating_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, d_model // 2),
            nn.GELU(), 
            nn.LayerNorm(d_model // 2), 
            nn.Linear(d_model // 2, 2)  
        ) 
        
        self.out_dim = d_model

    def forward(self, v_item, v_text):

        v_item_proj = self.proj_item(v_item)       # [B, S, D_model]
        v_text_proj = self.proj_text(v_text)       # [B, S, D_model]

        gate_input = torch.cat([v_item, v_text], dim=-1)   # [B, S, D_total]
        gates = self.gating_mlp(gate_input)                # [B, S, 2]
        weights = F.softmax(gates, dim=-1)                 # [B, S, 2]

        h_seq = (
            weights[..., 0].unsqueeze(-1) * v_item_proj +
            weights[..., 1].unsqueeze(-1) * v_text_proj
        ) # [B, S, D_model]
        
        return h_seq

class UserTower(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, max_seq_len=50):
        super().__init__()

        self.fusion_layer = GatingFusionLayer(
            item_emb_dim=256, 
            text_emb_dim=256, 
            d_model=d_model
        )      
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            activation="gelu",
            dropout=0.3
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 

        self.final_proj = nn.Linear(d_model, d_model)

    def forward(self, batch):
        device = next(self.parameters()).device

        v_item = batch["item_embs"].to(device)         
        v_text = batch["review_embs"].to(device)             
        mask = batch["transformer_mask"].to(device)            

        h_seq = self.fusion_layer(v_item, v_text) # [B, S, D_model]

        h_seq_pos = self.pos_encoder(h_seq.permute(1, 0, 2)).permute(1, 0, 2)
        transformer_out = self.transformer_encoder(h_seq_pos, src_key_padding_mask=mask)

        mask_expanded = ~mask.unsqueeze(-1) 
        sum_pooled = (transformer_out * mask_expanded).sum(dim=1) # [B, D_model]
        count_non_pad = mask_expanded.sum(dim=1).clamp(min=1e-9)  
        
        user_embedding = sum_pooled / count_non_pad
        
        return F.normalize(self.final_proj(user_embedding), dim=-1)