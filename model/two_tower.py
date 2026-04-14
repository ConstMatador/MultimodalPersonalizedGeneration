# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, item_emb_matrix, out_dim=128, train_item_proj=True):
        super().__init__()
        num_items, in_dim = item_emb_matrix.shape
        self.item_table = nn.Embedding(num_items, in_dim, padding_idx=0)
        self.item_table.weight.data.copy_(torch.tensor(item_emb_matrix))
        self.item_table.weight.requires_grad = False  # 冻结离线item向量

        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        if not train_item_proj:
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, item_idx):
        x = self.item_table(item_idx)   # [..., in_dim]
        v = self.proj(x)                # [..., out_dim]
        return F.normalize(v, dim=-1)


class UserTower(nn.Module):
    def __init__(self, num_users, d=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 32)
        self.mlp = nn.Sequential(
            nn.Linear(d * 2 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, d)
        )

    def masked_mean(self, x, m):
        m = m.unsqueeze(-1)             # [B,L,1]
        s = (x * m).sum(dim=1)          # [B,D]
        c = m.sum(dim=1).clamp(min=1e-6)
        return s / c

    def forward(self, uid, hpos_vec, mpos, hneg_vec, mneg):
        hp = self.masked_mean(hpos_vec, mpos)
        hn = self.masked_mean(hneg_vec, mneg)
        u0 = self.user_emb(uid)
        u = self.mlp(torch.cat([hp, hn, u0], dim=-1))
        return F.normalize(u, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, item_tower, user_tower):
        super().__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower

    def bpr_loss(self, u, vp, vn):
        sp = (u * vp).sum(dim=-1)
        sn = (u * vn).sum(dim=-1)
        return -torch.log(torch.sigmoid(sp - sn) + 1e-8).mean()
