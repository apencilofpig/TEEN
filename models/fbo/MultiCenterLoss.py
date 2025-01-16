import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCenterLoss(nn.Module):
    def __init__(self, target_class, model):
        super(MultiCenterLoss, self).__init__()
        self.target_class = target_class
        self.model = model  # centers是多中心原型，类型为nn.Parameter

    def forward(self, x_feature, train_label):
        mask = train_label == self.target_class
        if torch.any(mask):
            # Extract features for the target class samples
            target_features = x_feature[mask]
            target_labels = train_label[mask]

            if 'cos' in self.model.mode:
                if self.model.dropout_fn is None:
                    new_logits = F.linear(F.normalize(target_features, p=2, dim=-1), F.normalize(self.model.centers.weight, p=2, dim=-1))
                else:
                    new_logits = F.linear(self.model.dropout_fn(F.normalize(target_features, p=2, dim=-1)), F.normalize(self.model.centers.weight, p=2, dim=-1))

                new_logits = self.model.args.temperature * new_logits

            elif 'dot' in self.model.mode:
                new_logits = self.model.centers(target_features)

            with torch.no_grad():
                if self.model.dropout_fn is None:
                    new_logits2 = F.linear(F.normalize(target_features, p=2, dim=-1), F.normalize(self.model.centers.weight, p=2, dim=-1))
                else:
                    new_logits2 = F.linear(self.model.dropout_fn(F.normalize(target_features, p=2, dim=-1)), F.normalize(self.model.centers.weight, p=2, dim=-1))

                new_logits2 = self.model.args.temperature * new_logits2
                _, new_labels = torch.max(new_logits2, dim=1)

            loss = F.cross_entropy(new_logits, new_labels)

            return loss

        else:
            return 0

    
    def assign_new_labels(self, x_feature, centers):
        # Step 1: 归一化样本特征和中心向量
        x_feature_normalized = F.normalize(x_feature, p=2, dim=-1)  # [batch_size, num_features]
        centers_normalized = F.normalize(centers, p=2, dim=-1)      # [num_centers, num_features]
        
        # Step 2: 计算余弦相似距离
        cosine_similarity = torch.matmul(x_feature_normalized, centers_normalized.T)  # [batch_size, num_centers]
        cosine_distance = 1 - cosine_similarity  # 转化为距离，越小表示越相似
        
        # Step 3: 找到每个样本距离最小的中心
        new_labels = cosine_distance.argmin(dim=1)  # 返回距离最小的中心索引，即新的标签
        
        return new_labels
