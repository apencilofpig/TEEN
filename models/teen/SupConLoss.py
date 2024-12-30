import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, normal_label):
        # Normalize features for cosine similarity
        normalized_features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(normalized_features, normalized_features.T) / self.temperature
        
        # Mask for positive pairs (normal samples)
        normal_mask = (labels == normal_label).unsqueeze(1) & (labels == normal_label).unsqueeze(0)
        
        # Mask for negative pairs (normal vs abnormal)
        abnormal_mask = (labels == normal_label).unsqueeze(1) & (labels != normal_label).unsqueeze(0)
        
        # Positive log probabilities
        positive_log_prob = torch.exp(similarity_matrix) * normal_mask
        positive_loss = -torch.log(positive_log_prob.sum(1) + 1e-8)
        
        # Negative log probabilities
        negative_log_prob = torch.exp(-similarity_matrix) * abnormal_mask
        negative_loss = -torch.log(negative_log_prob.sum(1) + 1e-8)
        
        return positive_loss.mean() + negative_loss.mean()