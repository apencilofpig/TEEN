import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, logits, targets, tau, margin):
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
        numerator = torch.exp(tau * (positive_logits - margin))

        # 计算分母
        negative_logits = logits.clone()
        negative_logits[range(batch_size), targets.long()] = -float('inf')
        denominator = numerator + torch.sum(torch.exp(tau * negative_logits), dim=1)

        # 计算损失
        loss = -torch.log(numerator / denominator).mean()

        return loss

def margin_loss(logits, targets, tau=16, margin=0.2):
    """
    Margin Loss 函数
    参数:
        logits (torch.Tensor): 模型输出的 logits, 形状为 (batch_size, num_classes)
        targets (torch.Tensor): 真实标签, 形状为 (batch_size,)
    返回:
        loss (torch.Tensor): 损失值
    """
    batch_size = logits.size(0)
    
    # 计算分母
    exp_logits = torch.exp(logits)
    sum_exp_logits = torch.sum(exp_logits, dim=1) - exp_logits[range(batch_size), targets]
    
    # 计算分子
    exp_logits = torch.exp(logits - tau * margin)
    target_exp_logits = exp_logits[range(batch_size), targets]
    
    # 计算损失
    loss = -torch.log(target_exp_logits / (sum_exp_logits + target_exp_logits))
    loss = torch.mean(loss)
    
    return loss