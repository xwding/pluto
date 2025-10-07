'''
Name: 
Date: 2025-09-27 22:43:13
Creator: 
Description: 
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ESDFCollisionLoss(nn.Module):
    def __init__(
        self,
        num_circles=3,
        ego_width=2.297,
        ego_front_length=4.049,
        ego_rear_length=1.127,
        resolution=0.2,
    ) -> None:
        super().__init__()
        # 车辆总长（米），由车体参考点向前的前悬长度和向后的后悬长度相加得到。
        ego_length = ego_front_length + ego_rear_length
        # 将车辆纵向按等距切分为 num_circles 段后，每段的长度（米）。这些段的中点将用来放置近似圆的圆心。
        interval = ego_length / num_circles
        # 用多少个圆沿纵向近似车身轮廓
        self.N = num_circles
        # 车辆宽度（米），用于确定每个圆需要覆盖的横向尺寸。
        self.width = ego_width
        self.length = ego_length
        # 车辆参考点到后保险杠（或车尾）方向的长度（米）。与 offset 的基点位置相关
        self.rear_length = ego_rear_length
        # ESDF 网格分辨率（米/像素），用于米↔像素坐标变换及安全半径的微调。
        self.resolution = resolution
        #  近似圆的安全半径（米）。它等于覆盖一个“宽为 ego_width、长为 interval”的矩形所需的最小圆半径（该矩形对角线的一半），再减去一个 resolution 的裕量：
        self.radius = math.sqrt(ego_width**2 + interval**2) / 2 - resolution
        # 每个近似圆心相对于车辆参考点沿车辆前向的纵向位移（米）。取值从“车尾方向的第一个段中心”开始，等间距到“车头方向的最后一个段中心”结束：
        self.offset = torch.Tensor(
            [-ego_rear_length + interval / 2 * (2 * i + 1) for i in range(num_circles)]
        )

    def forward(self, trajectory: Tensor, sdf: Tensor):
        """
        trajectory: (bs, T, 4) - [x, y, cos0, sin0]
        sdf: (bs, H, W)
        """
        bs, H, W = sdf.shape

        origin_offset = torch.tensor([W // 2, H // 2], device=sdf.device)
        offset = self.offset.to(sdf.device).view(1, 1, self.N, 1)
        # (bs, T, N, 2)
        # centers = pos + offset * heading
        centers = trajectory[..., None, :2] + offset * trajectory[..., None, 2:4]
        #  圆心从米坐标转换为像素坐标；x/rez → 列，-y/rez → 行（y 取负以匹配图像坐标向下为正）。
        pixel_coord = torch.stack(
            [centers[..., 0] / self.resolution, -centers[..., 1] / self.resolution],
            dim=-1,
        )
        # 将像素坐标归一化到约 [-1, 1] 注: grid_sample 期望网格坐标在 [-1, 1]
        grid_xy = pixel_coord / origin_offset
        # 含义: 网格坐标是否落在有效采样范围内（留 0.95 的边界缓冲），超界将被屏蔽
        valid_mask = (grid_xy < 0.95).all(-1) & (grid_xy > -0.95).all(-1)
        on_road_mask = sdf[:, H // 2, W // 2] > 0

        # (bs, T, N)
        # 在 grid_xy 位置对 ESDF 进行双线性插值采样得到的距离值（米）
        distance = F.grid_sample(
            sdf.unsqueeze(1), grid_xy, mode="bilinear", padding_mode="zeros"
        ).squeeze(1)

        #仅当 cost > 0（即距离小于安全半径，存在碰撞风险）且掩码有效时参与损失；其余位置置零，不产生梯度
        cost = self.radius - distance
        valid_mask = valid_mask & (cost > 0) & on_road_mask[:, None, None]
        cost.masked_fill_(~valid_mask, 0)
        # 等价于对所有有效点的正部 ReLU 累加并归一化：
        # loss=∑b,t,nmax⁡(0,  r−dbtn)∑b,t,n1[(b,t,n) 有效]+10−6loss=∑b,t,n​1[(b,t,n) 有效]+10−6∑b,t,n​max(0,r−dbtn​)​
        # 其中 rr 为安全半径，dbtndbtn​ 为 ESDF 距离采样值
        loss = F.l1_loss(cost, torch.zeros_like(cost), reduction="none").sum(-1)
        loss = loss.sum() / (valid_mask.sum() + 1e-6)

        return loss
