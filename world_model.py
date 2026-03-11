import torch
import wandb
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from LSTM_discrete_embeddings import DiscreteDynamicsLSTM
from ArcBallDataset import ArcBallDatasetDiscrete, get_bin_configs

class WorldModel:
    def __init__(self, state_bin_cfg, delta_bin_cfg, lr=1e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_bin_cfg = state_bin_cfg 
        self.delta_bin_cfg = delta_bin_cfg
        
        self.model = DiscreteDynamicsLSTM(
            state_bins_list=state_bin_cfg, 
            delta_bins_list=delta_bin_cfg
        ).to(self.device)
        
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Scheduler Logic
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, 
            mode='min', 
            factor=0.5, 
            patience=1, 
            min_lr=1e-6
        )

    def train_step(self, batch):
        self.model.train()
        s_raw, a_raw, y_raw = batch
        
        # Move to device
        s, a, y = s_raw.to(self.device), a_raw.to(self.device), y_raw.to(self.device)
        
        self.opt.zero_grad()
        logits, _ = self.model(s, a)
        loss = self.model.loss_fn(logits, y)
        loss.backward()
        
        # Gradient clipping prevents exploding gradients in LSTMs
        #grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        grad_norm = 0
        
        self.opt.step()
        return loss.item(), grad_norm
    
    def step_scheduler(self, val_loss):
        """
        Updates the learning rate based on validation loss.
        Returns the new LR for logging purposes.
        """
        self.scheduler.step(val_loss)
        return self.opt.param_groups[0]['lr']

    def save_checkpoint(self, path="best_model.pth"):
        payload = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state_bin_cfg': self.state_bin_cfg,
            'delta_bin_cfg': self.delta_bin_cfg,
            'bin_edges': self.bin_edges
        }
        torch.save(payload, path)

# --- Execution Logic ---
if __name__ == "__main__":
    wandb.init(project="arcball_500_fixed_bins")
    ds_path = "/home/snath/data/dataset/arcball_discrete_600k.h5"
    configs = get_bin_configs(ds_path, target_bins=200)

    train_ds = ArcBallDatasetDiscrete(ds_path, mode='train', bin_configs=configs)
    val_ds = ArcBallDatasetDiscrete(ds_path, mode='val', bin_configs=configs)
    
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    wm = WorldModel(
        state_bin_cfg=train_ds.get_state_cfg(), 
        delta_bin_cfg=train_ds.get_bin_cfg()
    )
    wm.bin_edges = train_ds.get_bin_edges()

    best_val_loss = float('inf')
    patience = 7
    counter = 0

    for epoch in range(100):
        # 1. Training Phase
        epoch_train_loss, grad_norms = 0, []
        for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]"):
            loss_val, g_norm = wm.train_step(batch)
            epoch_train_loss += loss_val
            grad_norms.append(g_norm)
        
        avg_train_loss = epoch_train_loss / len(loader)
        
        # 2. Validation Phase
        wm.model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                s, a, y = [b.to(wm.device) for b in batch]
                logits, _ = wm.model(s, a)
                v_loss += wm.model.loss_fn(logits, y).item()
        avg_val_loss = v_loss / len(val_loader)
        
        # 3. Scheduler Step (Annealing Logic)
        current_lr = wm.step_scheduler(avg_val_loss)
        
        # Logging
        wandb.log({
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss, 
            "lr": current_lr,
            "grad_norm_avg": np.mean(grad_norms)
        })
        
        print(f"Epoch {epoch}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | LR {current_lr:.2e}")

        # 4. Early Stopping & Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wm.save_checkpoint("best_model.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break