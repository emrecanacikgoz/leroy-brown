import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageEncoder(pl.LightningModule):
    def __init__(
            self, 
            in_features: int=15,
            task_embedding_size: int=64, 
            hidden_size: int=64, 
            latent_language_encoder_features: int=64, 
            num_tasks: int=34, 
            l2_normalize_language_embeddings: bool=False,
        ):
        super().__init__()

        self.embed = nn.Embedding(num_tasks, task_embedding_size)
        self.l2_normalize_output = l2_normalize_language_embeddings
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features+task_embedding_size, out_features=hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            #nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(in_features=hidden_size, out_features=latent_language_encoder_features),
        )
        #self.fc_in = nn.Linear(in_features=in_features+task_embedding_size, out_features=latent_language_encoder_features)

    def forward(self, state: torch.tensor, start_state: torch.tensor, task_id: torch.tensor):
        assert len(state.shape) == 3 and len(start_state.shape) == 3 and len(task_id.shape) == 1
        T = state.shape[1]

        # embed task-ids and initial-states
        task_id = self.embed(task_id)
        
        conditional_inputs = torch.cat([task_id.unsqueeze(1), start_state], dim=-1) 
        embed_l = self.mlp(conditional_inputs)

        if self.l2_normalize_output:
            embed_l = F.normalize(embed_l, p=2, dim=1)

        embed_l = embed_l.repeat(1, T, 1)

        return embed_l