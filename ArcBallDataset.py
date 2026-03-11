import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

def get_bin_configs(path, target_bins=200, train_ratio=0.8):
    """
    Calculates bin edges based solely on the training portion of raw data.
    This is the GROUND TRUTH for your model's architecture.
    """
    with h5py.File(path, 'r') as f:
        data = f['dataset'][:]
    
    N = data.shape[0]
    train_end = int(N * train_ratio)
    
    s_raw_train = data[:train_end, 0:4]
    d_raw_train = data[:train_end, 5:9] - data[:train_end, 0:4]
    
    bin_configs = {'state_edges': [], 'delta_edges': []}
    quantile_range = np.linspace(0, 1, target_bins + 1)

    for i in range(4):
        s_edges = np.unique(np.quantile(s_raw_train[:, i], quantile_range))
        d_edges = np.unique(np.quantile(d_raw_train[:, i], quantile_range))
        
        bin_configs['state_edges'].append(s_edges)
        bin_configs['delta_edges'].append(d_edges)
        
    return bin_configs

class ArcBallDatasetDiscrete(Dataset):
    def __init__(self, path, mode='train', seq_len=20, bin_configs=None):
        self.seq_len = seq_len
        self.edges = bin_configs # Ground truth from get_bin_configs
        
        if self.edges is None:
            raise ValueError("bin_configs must be provided to ensure consistent discretization.")

        with h5py.File(path, 'r') as f:
            data = f['dataset'][:] 

        N = data.shape[0]
        train_end = int(N * 0.8)
        val_end = int(N * 0.9)
        slc = {'train': slice(0, train_end), 'val': slice(train_end, val_end), 'test': slice(val_end, None)}[mode]
        
        data_mode = data[slc]
        s_raw = data_mode[:, 0:4]
        a_raw = data_mode[:, 4].astype(int)
        d_raw = data_mode[:, 5:9] - s_raw

        num_samples = data_mode.shape[0]
        self.s_idx_np = np.zeros((num_samples, 4), dtype=np.int64)
        self.y_idx_np = np.zeros((num_samples, 4), dtype=np.int64)
        
        # --- THE FIX: FORCED DISCRETIZATION ---
        for i in range(4):
            se = self.edges['state_edges'][i]
            de = self.edges['delta_edges'][i]
            
            # The model architecture depends on these counts:
            num_s_bins = len(se) - 1 
            num_d_bins = len(de) - 1

            # np.digitize returns indices from 0 to len(edges).
            # We must clip to [0, num_bins - 1] so it NEVER exceeds 
            # the embedding/linear layer dimensions.
            self.s_idx_np[:, i] = np.clip(np.digitize(s_raw[:, i], se[1:-1]), 0, num_s_bins - 1)
            self.y_idx_np[:, i] = np.clip(np.digitize(d_raw[:, i], de[1:-1]), 0, num_d_bins - 1)

        self.a_oh_np = np.zeros((num_samples, 3), dtype=np.float32)
        self.a_oh_np[np.arange(num_samples), np.clip(a_raw, 0, 2)] = 1.0

        self.s_idx = torch.tensor(self.s_idx_np, dtype=torch.long)
        self.a_oh = torch.tensor(self.a_oh_np, dtype=torch.float32)
        self.y_idx = torch.tensor(self.y_idx_np, dtype=torch.long)

    def __len__(self):
        return len(self.s_idx) - self.seq_len + 1

    def __getitem__(self, idx):
        return (self.s_idx[idx : idx + self.seq_len], 
                self.a_oh[idx : idx + self.seq_len], 
                self.y_idx[idx : idx + self.seq_len])

    def get_bin_cfg(self):
        return [len(e) - 1 for e in self.edges['delta_edges']]

    def get_state_cfg(self):
        return [len(e) - 1 for e in self.edges['state_edges']]
    
    def get_bin_edges(self):
        return self.edges