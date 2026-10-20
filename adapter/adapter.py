import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self,hidden_size : int, rank:int, dropout:float, scale:float=1.0):
        super().__init__()
        self.w_down = nn.Linear(hidden_size,rank)
        self.w_up = nn.Linear(rank,hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def __init_ide(self):
        nn.init.normal_(self.w_down.weight,std=0.02)
        nn.init.zeros_(self.w_down.bias)
        nn.init.zeros_(self.w_up.weight)
        nn.init.zeros_(self.w_up.bias)

    def forward(self,x):
        return x + self.scale * self.w_up(self.dropout(self.act(self.w_down(x))))


class LinearWithAdapter(nn.Module):
    def __init__(self,base:nn.Linear,rank:int,dropout:float):
        super().__init__()
        self.base = base
        self.adapter = BottleneckAdapter(self.base.out_features,rank = rank , dropout= dropout)

    def forward(self,x):
        return self.adapter(self.base(x))