import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, tau=1.0, margin=0.5):
        super(MarginLoss, self).__init__()
        self.tau = tau
        self.margin = margin

    def forward(self, logits, targets):
        """
        计算基于间距的损失函数
        
        参数:
        logits (torch.Tensor): 模型的输出logits,shape为(batch_size, num_classes)
        targets (torch.Tensor): 真实类别标签,shape为(batch_size,)
        
        返回:
        loss (torch.Tensor): 损失值
        """
        batch_size, num_classes = logits.shape

        # 获取正类别的logits
        positive_logits = logits[range(batch_size), targets.long()]

        # 计算分子
        numerator = torch.exp(self.tau * (positive_logits - self.margin))

        # 计算分母
        negative_logits = logits.clone()
        negative_logits[range(batch_size), targets.long()] = -float('inf')
        denominator = numerator + torch.sum(torch.exp(self.tau * negative_logits), dim=1)

        # 计算损失
        loss = -torch.log(numerator / denominator).mean()

        return loss