import torch
import torch.nn as nn



class SideEncoder_Var(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.device = configs["device"]

        self.var_emb_linear = nn.Sequential(
            nn.Linear(configs['seq_len'], (configs['seq_len'] + configs["var_emb"])//2),
            nn.SiLU(),
            nn.Linear((configs['seq_len'] + configs["var_emb"])//2, configs["var_emb"]),
        )
        self.total_emb_dim = configs["var_emb"] + configs["time_emb"]

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x_raw, tp):
        B, L = tp.shape
        num_vars = x_raw.shape[2]
        time_emb = self.time_embedding(tp, self.configs["time_emb"])
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, num_vars, -1) #(b, seq_len, n_var, dim)
        var_emb = self.var_emb_linear(x_raw).expand(-1, L, -1, -1) #(B, 1, num_var, var_dim)
        side_emb = torch.cat([time_emb, var_emb], dim=-1)
        side_emb = side_emb.permute(0, 3, 2, 1)
        return side_emb