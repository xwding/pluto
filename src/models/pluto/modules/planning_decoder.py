from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding
from ..layers.mlp_layer import MLPLayer


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout) -> None:
        super().__init__()
        self.dim = dim

        self.r2r_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.m2m_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        m_pos: Optional[Tensor] = None,
    ):
        """
        tgt: (bs, R, M, dim)
        tgt_key_padding_mask: (bs, R)
        """
        bs, R, M, D = tgt.shape

        tgt = tgt.transpose(1, 2).reshape(bs * M, R, D)
        tgt2 = self.norm1(tgt)
        tgt2 = self.r2r_attn(
            tgt2, tgt2, tgt2, key_padding_mask=tgt_key_padding_mask.repeat(M, 1)
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt_tmp = tgt.reshape(bs, M, R, D).transpose(1, 2).reshape(bs * R, M, D)
        tgt_valid_mask = ~tgt_key_padding_mask.reshape(-1)
        tgt_valid = tgt_tmp[tgt_valid_mask]
        tgt2_valid = self.norm2(tgt_valid)
        tgt2_valid, _ = self.m2m_attn(
            tgt2_valid + m_pos, tgt2_valid + m_pos, tgt2_valid
        )
        tgt_valid = tgt_valid + self.dropout2(tgt2_valid)
        tgt = torch.zeros_like(tgt_tmp)
        tgt[tgt_valid_mask] = tgt_valid

        tgt = tgt.reshape(bs, R, M, D).view(bs, R * M, D)
        tgt2 = self.norm3(tgt)
        tgt2 = self.cross_attn(
            tgt2, memory, memory, key_padding_mask=memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm4(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt.reshape(bs, R, M, D)

        return tgt


class PlanningDecoder(nn.Module):
    def __init__(
        self,
        num_mode,
        decoder_depth,
        dim,
        num_heads,
        mlp_ratio,
        dropout,
        future_steps,
        yaw_constraint=False,
        cat_x=False,
    ) -> None:
        super().__init__()

        self.num_mode = num_mode
        self.future_steps = future_steps
        self.yaw_constraint = yaw_constraint
        self.cat_x = cat_x

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderLayer(dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_depth)
            ]
        )

        self.r_pos_emb = FourierEmbedding(3, dim, 64)
        self.r_encoder = PointsEncoder(6, dim)

        self.q_proj = nn.Linear(2 * dim, dim)

        self.m_emb = nn.Parameter(torch.Tensor(1, 1, num_mode, dim))
        self.m_pos = nn.Parameter(torch.Tensor(1, num_mode, dim))

        if self.cat_x:
            self.cat_x_proj = nn.Linear(2 * dim, dim)

        self.loc_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.yaw_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.vel_head = MLPLayer(dim, 2 * dim, self.future_steps * 2)
        self.pi_head = MLPLayer(dim, dim, 1)

        nn.init.normal_(self.m_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)

    def forward(self, data, enc_data):
        # B: batch size
        # R: 每个样本的参考线（候选路线）数量
        # P: 每条参考线被采样的点数
        # M: 模态数（num_mode，多模轨迹假设的数量）
        # D: 特征维度（decoder/encoder的隐藏维度）
        # Nmem: encoder 输出的“记忆”序列长度（场景/地图/目标的编码 token 数）
        
        # 包含已编码的目标、场景、地图等上下文特征。
        # 形状: (B, Nmem, D)
        # 用途: 作为 cross-attention 的 K/V；
        enc_emb = enc_data["enc_emb"]
        # 对 enc_emb 的 padding 掩码. True 表示该位置是 padding，需要在注意力中被忽略。
        # 形状: (B, Nmem) bool
        enc_key_padding_mask = enc_data["enc_key_padding_mask"]
        # 含义: 每条参考线按纵向采样的 2D 位置序列（通常是地图坐标或自车坐标系下的 x,y）。代码中会减去第一个点的 (x,y)，得到相对位移，增强平移不变性。
        # 形状: (B, R, P, 2)
        r_position = data["reference_line"]["position"]
        # 参考线上每个采样点的切向方向向量（dx, dy），通常为单位向量，表示参考线的局部方向。
        # 形状: (B, R, P, 2)
        r_vector = data["reference_line"]["vector"]
        # 参考线上每个采样点的朝向（航向角 yaw，弧度制）。代码里会转成 [cos(yaw), sin(yaw)] 以避免角度环形不连续问题。
        # 形状: (B, R, P)
        r_orientation = data["reference_line"]["orientation"]
        # 每条参考线每个点是否有效的掩码。True 表示该采样点存在/有效（比如未越界或数据可用）。
        # 形状: (B, R, P) bool
        r_valid_mask = data["reference_line"]["valid_mask"]
        r_key_padding_mask = ~r_valid_mask.any(-1)

        r_feature = torch.cat(
            [
                r_position - r_position[..., 0:1, :2],
                r_vector,
                torch.stack([r_orientation.cos(), r_orientation.sin()], dim=-1),
            ],
            dim=-1,
        )

        bs, R, P, C = r_feature.shape
        r_valid_mask = r_valid_mask.view(bs * R, P)
        r_feature = r_feature.reshape(bs * R, P, C)
        r_emb = self.r_encoder(r_feature, r_valid_mask).view(bs, R, -1)

        r_pos = torch.cat([r_position[:, :, 0], r_orientation[:, :, 0, None]], dim=-1)
        r_emb = r_emb + self.r_pos_emb(r_pos)

        # 每条参考线聚合后的特征。由参考线点云特征通过 PointsEncoder 编码得到，并叠加了参考线起点位置与朝向的 Fourier 位置编码，
        # 表示“这条路线”的语义与几何。 初始: (B, R, D) -->  (B, R, M, D)
        r_emb = r_emb.unsqueeze(2).repeat(1, 1, self.num_mode, 1)
        # 可学习的“模态 token”，为每个模态提供不同的先验/身份标识，使同一条参考线在不同模态下能产生不同的解码查询，从而支持多模轨迹。
        # 原始参数形状: (1, 1, M, D) -> (B, R, M, D)
        m_emb = self.m_emb.repeat(bs, R, 1, 1)
        # self.q_proj (B, R, M, 2D) --> (B, R, M, D)
        q = self.q_proj(torch.cat([r_emb, m_emb], dim=-1))

        for blk in self.decoder_blocks:
            q = blk(
                q,
                enc_emb,
                tgt_key_padding_mask=r_key_padding_mask,
                memory_key_padding_mask=enc_key_padding_mask,
                m_pos=self.m_pos,
            )
            assert torch.isfinite(q).all()

        if self.cat_x:
            x = enc_emb[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, R, self.num_mode, 1)
            q = self.cat_x_proj(torch.cat([q, x], dim=-1))
        # B: batch size
        # R: 每个样本的参考线（候选路线）数
        # M: 模态数（num_mode，多模轨迹假设的数量）
        # T: future_steps（未来时间步数）
        # D: 隐藏维度

        # 含义: 每个“参考线-模态”的未来位置预测序列。最后一维 2 表示平面坐标 (x, y)。
        # 坐标系: 与训练标签一致，通常是以参考线起点为原点/朝向对齐的本地坐标系（代码前面构造参考线特征时已做相对化处理）。
        # 形状: (B, R, M, T, 2)
        # 单位: 米（常见设定）
        loc = self.loc_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        
        yaw = self.yaw_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        # 含义: 未来速度向量分量。最后一维 2 表示速度在平面坐标系的两个分量 (vx, vy)。
        # 坐标系: 与 loc 对齐的同一坐标系（通常是局部参考线坐标系或数据集定义的车体/世界坐标系）。
        # 形状: (B, R, M, T, 2)
        # 单位: 米/秒（常见设定）
        vel = self.vel_head(q).view(bs, R, self.num_mode, self.future_steps, 2)
        # 含义: 混合权重的未归一化分数（logit）。每个“参考线-模态”对应一个标量，用于表征该候选轨迹的置信度/概率。
        # 形状: (B, R, M)
        # 用法: 通常在 M 维（每条参考线内）做 softmax 得到各模态概率；有的实现也会在 R×M 维度上统一 softmax 来筛选全局最佳候选。   
        pi = self.pi_head(q).squeeze(-1)

        traj = torch.cat([loc, yaw, vel], dim=-1)

        return traj, pi
