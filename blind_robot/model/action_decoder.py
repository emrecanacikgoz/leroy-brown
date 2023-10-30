import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from blind_robot.model.encoders.goal_encoder import StateGoalEncoder
from blind_robot.model.encoders.language_encoder import LanguageEncoder
from blind_robot.model.utils.positional_encoder import PositionalEncoding


class ActionDecoder(pl.LightningModule):
    def __init__(
        self,
        n_input=15,
        context_size=32,
        n_heads=4,
        n_layer=4,
        hidden_size=128,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        num_tasks=34,
        embedding_size=128,
        output_vocabs=None,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.output_vocabs = output_vocabs
        self.discrete_actions = output_vocabs is not None

        self.register_buffer(
            "mask",
            torch.triu(
                torch.full((context_size, context_size), float("-inf")), diagonal=1
            ),
        )

        self.fc_in = nn.Linear(
            in_features=n_input+embedding_size, out_features=hidden_size
        )
        self.pos = PositionalEncoding(
            d_model=hidden_size, dropout=dropout, max_len=context_size, batch_first=True
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer, num_layers=n_layer
        )

        if self.discrete_actions:
            self.fc_out = nn.Linear(in_features=hidden_size, out_features=(len(output_vocabs[0]) - 1) * 7)
        else:
            self.fc_out = nn.Linear(in_features=hidden_size, out_features=7)

    def forward(self, state, conditional_embeddings):
        assert len(state.shape) == 3
        
        if conditional_embeddings.shape[1] != state.shape[1]:
            conditional_embeddings = conditional_embeddings.repeat(1, state.shape[1], 1)
        states = torch.cat([state, conditional_embeddings], dim=-1) # TODO: can we do this inside StateGoalEncoder?

        x = self.fc_in(states)
        x = self.pos(x)

        mask = self.mask[: x.shape[1], : x.shape[1]]
        x = self.transformer(x, mask=mask, is_causal=True)

        actions = self.fc_out(x)

        if self.discrete_actions:
            actions = actions.view(actions.shape[0], actions.shape[1], 7, len(self.output_vocabs[0])-1)

        return actions

    @torch.no_grad()
    def step(self, state, contitional_embeddings):
        # print(state.shape)
        action_pred = self(state, contitional_embeddings)
        if self.discrete_actions:
            # un-digitize the actions by finding closest cutoff in output_vocab
            action_pred = action_pred.argmax(dim=-1, keepdim=False)
        assert len(action_pred.shape) == 3
        action_pred = action_pred[0, -1, :]
        if self.discrete_actions:
            action_pred_float = action_pred.clone().float()
            for i in range(7):
                code = action_pred[i].item()
                action_pred_float[i] = float(np.random.uniform(self.output_vocabs[i][code], self.output_vocabs[i][code + 1]))
            action_pred = action_pred_float
        else:
            action_pred[6] = torch.sigmoid(action_pred[6])

        if action_pred[6] > 0.5:
            action_pred[6] = 1
        else:
            action_pred[6] = -1
        return action_pred

