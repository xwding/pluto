# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import List, Optional

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # continuous_inputs.unsqueeze(-1) 把形状从 (..., D) 变成 (..., D, 1)
        # 与 self.freqs.weight（D, F）按最后两维做广播相乘，得到形状 (..., D, F)
        # 乘以 2π 是标准 Fourier 特征：2π f x
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        # 含义：每个维度的特征现在是 [cos(2π f_k x), sin(2π f_k x), x] 的拼接。保留原始输入 x 作为低频/恒等通道有助于模型捕捉非周期性或低频信息
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            # x[..., i, :] 取出第 i 个输入维度的特征向量，形状 (..., 2F + 1)
            # 把它喂给第 i 个 MLP：self.mlps[i]，输出形状一般是 (..., H)，H 是隐藏维/特征维
            # 这样对每个输入维度单独建模，相当于“按维度”的专家/通道处理
            continuous_embs[i] = self.mlps[i](x[..., i, :])
        # 将 D 个张量堆叠成形状 (D, ..., H)，然后在维度 0 上求和，得到形状 (..., H)
        # 含义：把各维度的特征贡献做一个加性聚合（类似 Deep Sets 的 sum 聚合，顺序无关且稳定）
        x = torch.stack(continuous_embs).sum(dim=0)
        # 最终通过一个输出头（线性层等）把 (..., H) 投到目标维度，返回 (..., O)
        return self.to_out(x)
