import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from dataloader.swat.swat import *

# 设置随机种子确保实验可重复
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 创建SWaT数据集类
class SWaTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 数据加载和预处理
def load_swat_dataset(base_classes=16, new_classes=20, feature_dim=51):
    """
    加载SWaT数据集并进行预处理
    
    base_classes: 基类数量
    new_classes: 新类数量
    feature_dim: 特征维度
    
    返回:
    - 基类数据和标签
    - 新类数据和标签
    """
    # 这里应该包含实际加载SWaT数据集的代码
    # 由于没有实际的数据集，我们模拟一些数据
    
    print("Loading SWaT dataset...")
    df = pd.read_csv('data/swat/data_newlabel.csv')
    inputs = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    inputs, labels = restraint_samples_number(inputs, labels, 3000)
    
    # 模拟数据集 (实际应用中替换为真实数据集的加载代码)
    # 基类数据 - 每类样本量较多
    # X_base = np.random.randn(base_classes * 100, feature_dim)
    # y_base = np.array([i for i in range(base_classes) for _ in range(100)])
    _, X_base, y_base = get_class_items(inputs, labels, range(base_classes))

    # 新类数据 - 每类样本量较少
    # X_new = np.random.randn(new_classes * 20, feature_dim)
    # y_new = np.array([i + base_classes for i in range(new_classes) for _ in range(20)])
    _, X_new, y_new = get_class_items(inputs, labels, range(base_classes, base_classes + new_classes))
    
    # 数据标准化
    scaler = StandardScaler()
    X_base = scaler.fit_transform(X_base)
    X_new = scaler.transform(X_new)
    
    # 转换为PyTorch张量
    X_base = torch.FloatTensor(X_base)
    y_base = torch.LongTensor(y_base)
    X_new = torch.FloatTensor(X_new)
    y_new = torch.LongTensor(y_new)
    
    print(f"Base classes data shape: {X_base.shape}, Labels shape: {y_base.shape}")
    print(f"New classes data shape: {X_new.shape}, Labels shape: {y_new.shape}")
    
    return (X_base, y_base), (X_new, y_new)

# 定义神经网络模型
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
    def forward(self, x):
        return self.layers(x)

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class IntrusionDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(IntrusionDetectionModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.classifier = Classifier(hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features
    
    # 得到logits而不是softmax结果
    def get_logits(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

class MergeClassifier(nn.Module):
    def __init__(self, total_classes):
        super(MergeClassifier, self).__init__()
        self.fc1 = nn.Linear(total_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, total_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, logits):
        x = F.relu(self.fc1(logits))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 元学习算法实现
class MAML:
    def __init__(self, model, alpha=0.01, beta=0.001, num_updates=5):
        self.model = model
        self.alpha = alpha  # 内循环学习率
        self.beta = beta   # 外循环学习率
        self.num_updates = num_updates
        self.criterion = nn.CrossEntropyLoss()
    
    def sample_task(self, X, y, num_classes, k_shot, q_shot):
        """
        从已有类别中采样任务
        X: 特征
        y: 标签
        num_classes: 任务中的类别数
        k_shot: 每个类的支持集样本数
        q_shot: 每个类的查询集样本数
        """
        # 获取所有可用类别
        available_classes = torch.unique(y).tolist()
        
        # 随机选择num_classes个类别
        selected_classes = random.sample(available_classes, num_classes)
        
        support_X, support_y, query_X, query_y = [], [], [], []
        
        for cls in selected_classes:
            # 获取当前类别的所有样本
            cls_idx = (y == cls).nonzero().view(-1)
            
            # 随机选择k_shot个样本作为支持集
            perm = torch.randperm(cls_idx.size(0))
            support_idx = cls_idx[perm[:k_shot]]
            
            # 随机选择q_shot个样本作为查询集
            query_idx = cls_idx[perm[k_shot:k_shot+q_shot]]
            
            support_X.append(X[support_idx])
            support_y.append(y[support_idx])
            query_X.append(X[query_idx])
            query_y.append(y[query_idx])
        
        # 合并支持集和查询集
        support_X = torch.cat(support_X)
        support_y = torch.cat(support_y)
        query_X = torch.cat(query_X)
        query_y = torch.cat(query_y)
        
        return support_X, support_y, query_X, query_y
    
    def adapt(self, support_X, support_y):
        """单步任务适应"""
        # 创建模型的临时副本
        temp_model = type(self.model)(*self.model.args, **self.model.kwargs)
        temp_model.load_state_dict(self.model.state_dict())
        temp_model.to(device)
        
        # 内循环优化
        optimizer = optim.Adam(temp_model.parameters(), lr=self.alpha)
        
        for _ in range(self.num_updates):
            logits, _ = temp_model(support_X)
            loss = self.criterion(logits, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return temp_model
    
    def meta_train(self, X, y, num_tasks, num_classes, k_shot, q_shot):
        """元训练过程"""
        meta_optimizer = optim.Adam(self.model.parameters(), lr=self.beta)
        
        for task_i in range(num_tasks):
            # 采样一个任务
            support_X, support_y, query_X, query_y = self.sample_task(
                X, y, num_classes, k_shot, q_shot)
            
            support_X, support_y = support_X.to(device), support_y.to(device)
            query_X, query_y = query_X.to(device), query_y.to(device)
            
            # 内循环适应
            adapted_model = self.adapt(support_X, support_y)
            
            # 外循环更新
            logits, _ = adapted_model(query_X)
            meta_loss = self.criterion(logits, query_y)
            
            # 计算梯度并更新原始模型
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            
            if (task_i + 1) % 100 == 0:
                print(f"Task {task_i+1}/{num_tasks}, Meta Loss: {meta_loss.item():.4f}")

# 模型合并方法实现
class ID_FSCIL:
    def __init__(self, input_dim, hidden_dim, base_classes, new_classes):
        """
        初始化ID-FSCIL模型
        
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        base_classes: 基类数量
        new_classes: 新类数量
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.base_classes = base_classes
        self.new_classes = new_classes
        self.total_classes = base_classes + new_classes
        
        # 初始化基类检测模型
        self.old_model = IntrusionDetectionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=base_classes
        )
        self.old_model.args = (input_dim, hidden_dim, base_classes)
        self.old_model.kwargs = {}
        
        # 初始化新类检测模型
        self.new_model = IntrusionDetectionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=new_classes
        )
        self.new_model.args = (input_dim, hidden_dim, new_classes)
        self.new_model.kwargs = {}
        
        # 初始化合并分类器
        self.merge_classifier = MergeClassifier(self.total_classes)
        
        # 将模型迁移到设备上
        self.old_model.to(device)
        self.new_model.to(device)
        self.merge_classifier.to(device)
        
        # 初始化MAML
        self.maml = MAML(self.new_model, alpha=0.01, beta=0.001, num_updates=5)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def initial_training(self, X_old, y_old, num_tasks=1000, num_classes=5, k_shot=6, q_shot=6):
        """
        初始训练阶段 - 训练旧类检测模型并进行元学习
        """
        print("Initial Training Phase")
        
        # 1. 训练旧类检测模型
        print("Training Old Class Detector...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_old, y_old, test_size=0.2, random_state=42)
        
        # 创建数据加载器
        train_dataset = SWaTDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 训练旧类检测模型
        optimizer = optim.Adam(self.old_model.parameters(), lr=0.001)
        
        for epoch in range(10):
            self.old_model.train()
            total_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                logits, _ = self.old_model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            # 计算平均损失
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
        
        # 评估旧类检测模型
        self.old_model.eval()
        X_test, y_test = X_test.to(device), y_test.to(device)
        with torch.no_grad():
            logits, _ = self.old_model(X_test)
            preds = torch.argmax(logits, dim=1)
            accuracy = torch.sum(preds == y_test).item() / len(y_test)
            print(f"Old Model Accuracy: {accuracy:.4f}")
        
        # 2. 元学习新类检测模型
        print("Meta-Training New Class Detector...")
        self.maml.meta_train(X_old, y_old, num_tasks, num_classes, k_shot, q_shot)
    
    def incremental_training(self, X_new, y_new, X_old_subset, y_old_subset, k_shot=6, q_shot=6):
        """
        增量训练阶段 - 合并旧类和新类检测模型
        """
        print("Incremental Training Phase")
        
        # 1. 微调新类检测模型
        print("Fine-tuning New Class Detector...")
        
        # 为每个新类选择k_shot个样本
        support_X_new, support_y_new = [], []
        test_X_new, test_y_new = [], []
        
        for cls in range(self.base_classes, self.base_classes + self.new_classes):
            # 获取当前类别的所有样本
            cls_idx = (y_new == cls).nonzero().view(-1)
            
            # 随机选择k_shot个样本作为支持集
            perm = torch.randperm(cls_idx.size(0))
            support_idx = cls_idx[perm[:k_shot]]
            
            # 其余样本作为测试集
            test_idx = cls_idx[perm[k_shot:]]
            
            support_X_new.append(X_new[support_idx])
            support_y_new.append(y_new[support_idx] - self.base_classes)  # 调整标签范围为[0, new_classes-1]
            
            if len(test_idx) > 0:
                test_X_new.append(X_new[test_idx])
                test_y_new.append(y_new[test_idx])
        
        # 合并支持集
        support_X_new = torch.cat(support_X_new).to(device)
        support_y_new = torch.cat(support_y_new).to(device)
        
        # 合并测试集（如果有）
        if test_X_new:
            test_X_new = torch.cat(test_X_new).to(device)
            test_y_new = torch.cat(test_y_new).to(device)
        
        # 微调新类检测模型
        optimizer = optim.Adam(self.new_model.parameters(), lr=0.01)
        
        for epoch in range(30):
            self.new_model.train()
            
            # 前向传播和反向传播
            logits, _ = self.new_model(support_X_new)
            loss = self.criterion(logits, support_y_new)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Fine-tuning Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
        
        # 2. 训练合并分类器
        print("Training Merge Classifier...")
        
        # 准备用于训练合并分类器的数据
        X_old_subset = X_old_subset.to(device)
        y_old_subset = y_old_subset.to(device)
        
        # 获取旧类和新类模型的logits
        with torch.no_grad():
            old_logits = self.old_model.get_logits(X_old_subset)
            new_logits_old = self.new_model.get_logits(X_old_subset)
            
            old_logits_new = self.old_model.get_logits(support_X_new)
            new_logits = self.new_model.get_logits(support_X_new)
        
        # 构建组合logits
        combined_logits_old = torch.cat([old_logits, torch.zeros(old_logits.size(0), self.new_classes).to(device)], dim=1)
        combined_logits_new = torch.cat([torch.zeros(new_logits.size(0), self.base_classes).to(device), new_logits], dim=1)
        
        # 合并数据
        merge_X = torch.cat([combined_logits_old, combined_logits_new])
        merge_y = torch.cat([y_old_subset, self.base_classes + support_y_new])
        
        # 创建数据加载器
        merge_dataset = SWaTDataset(merge_X, merge_y)
        merge_loader = DataLoader(merge_dataset, batch_size=32, shuffle=True)
        
        # 训练合并分类器
        optimizer = optim.Adam(self.merge_classifier.parameters(), lr=0.01)
        
        for epoch in range(50):
            self.merge_classifier.train()
            total_loss = 0
            
            for X_batch, y_batch in merge_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                logits = self.merge_classifier(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            # 计算平均损失
            avg_loss = total_loss / len(merge_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Merge Classifier Epoch {epoch+1}/50, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """
        使用合并模型进行预测
        """
        X = X.to(device)
        
        # 获取旧类和新类模型的logits
        with torch.no_grad():
            old_logits = self.old_model.get_logits(X)
            new_logits = self.new_model.get_logits(X)
            
            # 构建组合logits
            combined_logits = torch.cat([old_logits, new_logits], dim=1)
            
            # 使用合并分类器进行预测
            final_logits = self.merge_classifier(combined_logits)
            predictions = torch.argmax(final_logits, dim=1)
            
        return predictions.cpu().numpy()
    
    def evaluate(self, X, y):
        """
        评估模型性能
        """
        y_np = y.cpu().numpy()
        y_pred = self.predict(X)
        
        # 计算总体准确率
        accuracy = accuracy_score(y_np, y_pred)
        
        # 计算旧类的准确率
        old_idx = y_np < self.base_classes
        if np.any(old_idx):
            old_accuracy = accuracy_score(y_np[old_idx], y_pred[old_idx])
        else:
            old_accuracy = 0
        
        # 计算新类的准确率
        new_idx = y_np >= self.base_classes
        if np.any(new_idx):
            new_accuracy = accuracy_score(y_np[new_idx], y_pred[new_idx])
        else:
            new_accuracy = 0
        
        return {
            'total_accuracy': accuracy,
            'old_class_accuracy': old_accuracy,
            'new_class_accuracy': new_accuracy
        }

# 主函数
def main():
    # 设置参数
    input_dim = 51  # SWaT数据集的特征维度
    hidden_dim = 128
    base_classes = 16  # 基类数量
    new_classes = 20  # 新类数量
    k_shot = 6  # 每个类的支持集样本数
    
    # 加载数据集
    (X_base, y_base), (X_new, y_new) = load_swat_dataset(
        base_classes=base_classes, 
        new_classes=new_classes, 
        feature_dim=input_dim
    )
    
    # 初始化ID-FSCIL模型
    id_fscil = ID_FSCIL(input_dim, hidden_dim, base_classes, new_classes)
    
    # 初始训练阶段
    id_fscil.initial_training(
        X_base, y_base, 
        num_tasks=1000, 
        num_classes=5, 
        k_shot=k_shot, 
        q_shot=k_shot
    )
    
    # 抽取少量旧类样本用于增量训练
    # 为每个旧类随机选择k_shot个样本
    X_old_subset, y_old_subset = [], []
    
    for cls in range(base_classes):
        # 获取当前类别的所有样本
        cls_idx = (y_base == cls).nonzero().view(-1)
        
        # 随机选择k_shot个样本
        perm = torch.randperm(cls_idx.size(0))
        selected_idx = cls_idx[perm[:k_shot]]
        
        X_old_subset.append(X_base[selected_idx])
        y_old_subset.append(y_base[selected_idx])
    
    # 合并样本
    X_old_subset = torch.cat(X_old_subset)
    y_old_subset = torch.cat(y_old_subset)
    
    # 增量训练阶段
    id_fscil.incremental_training(
        X_new, y_new, 
        X_old_subset, y_old_subset, 
        k_shot=k_shot
    )
    
    # 评估模型
    print("\nEvaluating model...")
    
    # 合并所有数据进行评估
    X_all = torch.cat([X_base, X_new])
    y_all = torch.cat([y_base, y_new])
    
    # 随机抽样进行评估（避免内存问题）
    sample_size = min(5000, len(X_all))
    indices = torch.randperm(len(X_all))[:sample_size]
    X_eval = X_all[indices]
    y_eval = y_all[indices]
    
    # 评估整体性能
    results = id_fscil.evaluate(X_eval, y_eval)
    
    print(f"Total Accuracy: {results['total_accuracy']:.4f}")
    print(f"Old Class Accuracy: {results['old_class_accuracy']:.4f}")
    print(f"New Class Accuracy: {results['new_class_accuracy']:.4f}")

if __name__ == "__main__":
    main()