import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCenterLoss(nn.Module):
    def __init__(self, num_centers, feature_dim, centers):
        super(MultiCenterLoss, self).__init__()
        self.num_centers = num_centers
        self.centers = centers

    def forward(self, features, labels):
        # features: (batch_size, feature_dim)
        # labels: (batch_size,)
        # 初始化损失为0
        indices = (labels == 0).nonzero(as_tuple=True)[0]  # 使用 nonzero 查找索引
        features = features[indices]
        labels = labels[indices]
        loss = 0
        for i in range(len(features)):
            feature = features[i]

            distances = [F.pairwise_distance(feature.unsqueeze(0), center.unsqueeze(0)) for center in self.centers]
            min_distance = torch.min(torch.stack(distances))
            loss += min_distance
        return loss / (len(features) + 1e-5)
