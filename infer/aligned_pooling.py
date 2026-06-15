import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGuidedWeightedPooling(nn.Module):
    """Pool a multi-value id feature with weights learned from aligned floats."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.value_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _transform_values(self, values: torch.Tensor) -> torch.Tensor:
        # Keep the gate numerically stable when raw aligned statistics vary a lot.
        return torch.sign(values) * torch.log1p(torch.abs(values))

    def forward(
        self,
        ids: torch.Tensor,
        values: torch.Tensor,
        embedding: nn.Embedding,
    ) -> torch.Tensor:
        mask = (ids != 0).unsqueeze(-1).float()  # (B, L, 1)
        emb = embedding(ids.long())  # (B, L, D)

        gate_inp = self._transform_values(values).unsqueeze(-1)  # (B, L, 1)
        gate = torch.sigmoid(self.value_gate(gate_inp)) * mask

        denom = gate.sum(dim=1).clamp(min=1e-6)
        pooled = (emb * gate).sum(dim=1) / denom
        return pooled


class AlignedUserDenseGuidedToken(nn.Module):
    """Build one user token from aligned ``user_int`` / ``user_dense`` pairs."""

    def __init__(
        self,
        pair_specs: List[Dict[str, int]],
        emb_dim: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.pair_specs = pair_specs
        self._logged_dense_guided_usage = False

        self.embs = nn.ModuleDict({
            str(spec['fid']): nn.Embedding(int(spec['vocab_size']) + 1, emb_dim, padding_idx=0)
            for spec in pair_specs
        })
        self.poolers = nn.ModuleDict({
            str(spec['fid']): DenseGuidedWeightedPooling()
            for spec in pair_specs
        })
        self.out_proj = nn.Sequential(
            nn.Linear(len(pair_specs) * emb_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self._init_params()

    def _init_params(self) -> None:
        for emb in self.embs.values():
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0

    def forward(
        self,
        user_int_feats: torch.Tensor,
        user_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        pooled_list = []
        first_gate_stats = None
        for spec in self.pair_specs:
            int_offset = spec['int_offset']
            int_length = spec['int_length']
            dense_offset = spec['dense_offset']
            dense_length = spec['dense_length']

            ids = user_int_feats[:, int_offset:int_offset + int_length]
            values = user_dense_feats[:, dense_offset:dense_offset + dense_length]
            pooler = self.poolers[str(spec['fid'])]
            emb = self.embs[str(spec['fid'])]
            pooled = pooler(ids, values, emb)
            pooled_list.append(pooled)
            if first_gate_stats is None:
                valid_mask = (ids != 0)
                valid_count = int(valid_mask.sum().item())
                if valid_count > 0:
                    transformed = pooler._transform_values(values)
                    gate = torch.sigmoid(pooler.value_gate(transformed.unsqueeze(-1))).squeeze(-1)
                    mean_gate = float(gate[valid_mask].mean().item())
                else:
                    mean_gate = 0.0
                first_gate_stats = (spec['fid'], valid_count, mean_gate)

        cat = torch.cat(pooled_list, dim=-1)
        if not self._logged_dense_guided_usage and first_gate_stats is not None:
            fid, valid_count, mean_gate = first_gate_stats
            logging.info(
                "dense_guided_user_pooling in use: pair_count=%s, first_fid=%s, "
                "valid_positions=%s, mean_gate=%.4f",
                len(self.pair_specs), fid, valid_count, mean_gate)
            self._logged_dense_guided_usage = True
        return F.silu(self.out_proj(cat))
