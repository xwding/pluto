'''
Name: 
Date: 2025-09-27 22:43:13
Creator: 
Description: 
'''
import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder
from ..layers.fourier_embedding import FourierEmbedding


class MapEncoder(nn.Module):
    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        use_lane_boundary=False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.use_lane_boundary = use_lane_boundary
        self.polygon_channel = (
            polygon_channel + 4 if use_lane_boundary else polygon_channel
        )

        self.polygon_encoder = PointsEncoder(self.polygon_channel, dim)
        self.speed_limit_emb = FourierEmbedding(1, dim, 64)

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data) -> torch.Tensor:
        # （车道、车道连接、斑马线）所有位置与朝向都已在特征归一化阶段转换到“当前时刻自车后轴中心为原点、x 向前、y 向左”的自车坐标系，
        # 位置单位为米、角度单位为弧度。
        
        # polygon_center 的形状是 [bs, M, 3]，每个元素为 [x, y, yaw]，描述该多边形的代表点（通常是中心线中点）相对自车的位置与朝向
        polygon_center = data["map"]["polygon_center"]
        # polygon_type 为 [bs, M] 的整型索引，区分 LANE / LANE_CONNECTOR / CROSSWALK 三类
        polygon_type = data["map"]["polygon_type"].long()
        # polygon_on_route 为 [bs, M] 的 0/1 标记，表示该元素是否在规划路线之上；
        polygon_on_route = data["map"]["polygon_on_route"].long()
        # polygon_tl_status 为 [bs, M] 的整型信号灯状态（来自 nuPlan 的枚举，包含 UNKNOWN/RED/YELLOW/GREEN 等）；
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        # polygon_has_speed_limit 为 [bs, M] 的布尔标记，指示是否存在速度限制；
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        # polygon_speed_limit 为 [bs, M] 的浮点数，给出该多边形的限速值（单位 m/s），在 has_speed_limit 为 False 时该数值不生效，后续会用一个“未知限速”的嵌入向量代替
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        
        # 每个多边形沿三条边（索引 0: 中心线，1: 左边界，2: 右边界）均匀采样得到的逐点几何描述
        # point_position 的形状为 [bs, M, 3, P, 2]，存储每条边的 P 个采样点坐标；
        point_position = data["map"]["point_position"]
        # point_vector 的形状为 [bs, M, 3, P, 2]，是相邻离散点的位移向量（相当于沿边的切向量）
        point_vector = data["map"]["point_vector"]
        # point_orientation 的形状为 [bs, M, 3, P]，向量的切向朝向角。
        point_orientation = data["map"]["point_orientation"]
        # valid_mask 的形状为 [bs, M, 3, P]，给出每个多边形在中心线维度上各采样点是否落在兴趣区域/可用范围内的布尔掩码（为后续按点筛选与池化做准备）
        valid_mask = data["map"]["valid_mask"]

        if self.use_lane_boundary:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                    point_position[:, :, 1] - point_position[:, :, 0],
                    point_position[:, :, 2] - point_position[:, :, 0],
                ],
                dim=-1,
            )
        else:
            polygon_feature = torch.cat(
                [
                    point_position[:, :, 0] - polygon_center[..., None, :2],
                    point_vector[:, :, 0],
                    torch.stack(
                        [
                            point_orientation[:, :, 0].cos(),
                            point_orientation[:, :, 0].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
