import torch
import torch.nn as nn

class DiscreteDynamicsLSTM(nn.Module):
    def __init__(self, state_bins_list, delta_bins_list, embed_dim=64, hidden_dim=1128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(bins, embed_dim) for bins in state_bins_list
        ])
        
        self.action_fc = nn.Linear(3, embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim * 5, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, bins) for bins in delta_bins_list
        ])

        # Automatically apply init on creation
        self.apply_custom_init()

    def apply_custom_init(self):
        """Iterates through all modules and applies Xavier/Orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # Orthogonal is superior for recurrent weights to maintain gradient flow
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1.0 (the second quarter of the bias vector in PyTorch)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
            
            elif isinstance(m, nn.Embedding):
                # Small normal distribution is standard for embeddings
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, s_idx, a_oh, hc=None):
        s_embeds = torch.cat([self.embeddings[i](s_idx[:, :, i]) for i in range(4)], dim=-1)
        a_embed = self.action_fc(a_oh)
        x = torch.cat([s_embeds, a_embed], dim=-1)
        
        out, (h, c) = self.lstm(x, hc)
        out = self.dropout(out)
        logits = [head(out) for head in self.heads]
        return logits, (h, c)

    def loss_fn(self, logits, targets):
        loss = 0
        for i in range(4):
            loss += nn.functional.cross_entropy(
                logits[i].transpose(1, 2), 
                targets[:, :, i], 
                label_smoothing=0.1
            )
        return loss