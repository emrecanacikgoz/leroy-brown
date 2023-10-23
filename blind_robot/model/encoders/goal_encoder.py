import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateGoalEncoder(pl.LightningModule):
    def __init__(self, in_features: int=15, hidden_size: int=64, latent_goal_encoder_features: int=128, l2_normalize_goal_embeddings: bool=False):
        super().__init__()
        self.l2_normalize_output = l2_normalize_goal_embeddings
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=latent_goal_encoder_features),
        )
        #self.ln = nn.LayerNorm(goal_encoder_hidden_size)

    def forward(self, state, goal_state):
        assert len(state.shape) == 3 and len(goal_state.shape) == 3
        T = state.shape[1]
        goal_state = goal_state.repeat(1, T, 1)

        x = self.mlp(goal_state)
        
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)

        return x