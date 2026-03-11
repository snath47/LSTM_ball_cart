import torch
import numpy as np
from LSTM_discrete_embeddings import DiscreteDynamicsLSTM

class InferenceWorldModel:
    def __init__(self, checkpoint_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Fixed the pickle error for PyTorch 2.6+
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.state_bin_cfg = checkpoint['state_bin_cfg']
        self.delta_bin_cfg = checkpoint['delta_bin_cfg']
        self.bin_edges = checkpoint['bin_edges']
        
        self.model = DiscreteDynamicsLSTM(
            state_bins_list=self.state_bin_cfg, 
            delta_bins_list=self.delta_bin_cfg
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded. Device: {self.device}")

    def _discretize(self, continuous_state):
        """Converts float state to discrete bin indices."""
        indices = []
        for i in range(4):
            edges = self.bin_edges['state_edges'][i]
            idx = np.digitize(continuous_state[i], edges[1:-1])
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=self.device).view(1, 1, 4)

    @torch.no_grad()
    def get_step(self, raw_state, action_idx, hc=None):
        """
        Single-step transition for MPC.
        Returns: (predicted_next_state_float, next_hidden_state)
        """
        s_idx = self._discretize(raw_state)
        
        # Prepare action one-hot
        a_oh = torch.zeros((1, 1, 3), device=self.device)
        a_oh[0, 0, action_idx] = 1.0
        
        # Forward pass
        logits, next_hc = self.model(s_idx, a_oh, hc)
        
        # De-bin the predicted deltas
        predicted_next_state = []
        for i in range(4):
            # Get most likely delta bin
            d_idx = torch.argmax(logits[i][0, -1, :], dim=-1).item()
            edges = self.bin_edges['delta_edges'][i]
            
            # Midpoint de-binning
            delta_val = (edges[d_idx] + edges[d_idx+1]) / 2.0
            predicted_next_state.append(raw_state[i] + delta_val)
                
        return np.array(predicted_next_state), next_hc
    
if __name__ == "__main__":
    model = InferenceWorldModel("best_model.pth")
    raw_state = np.array([0.0, 0.0, 0.0, 0.0])  # Example state
    action_idx = 1  # Example action index
    predicted_next_state, _ = model.get_step(raw_state, action_idx)
    print("Predicted Next State:", predicted_next_state)