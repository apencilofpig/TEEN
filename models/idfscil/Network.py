import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.resnet18_encoder import *
import models.resnet18_swat
from models.resnet20_cifar import *
import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        if self.args.dataset in ['swat', 'wadi', 'hai']:
            # self.encoder = nn.Sequential(
            #     nn.Conv2d(1, 64, 3),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.3),
            #     # nn.Conv2d(32, 64, 3),
            #     # nn.BatchNorm2d(64),
            #     # nn.ReLU(),
            #     # nn.MaxPool2d(2, stride=2),
            #     # nn.Dropout(0.3),
            #     nn.Conv2d(64, 512, 3),
            #     nn.BatchNorm2d(512),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, stride=2),
            #     nn.Dropout(0.3)
            # )
            self.encoder = nn.Sequential(
                nn.Embedding(256, 32),
                models.resnet18_swat.resnet18(False, args)
            )
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.dropout_fn = nn.Dropout(0.3)

    def forward_metric(self, x):
        x_feature = self.encode(x)
        if 'cos' in self.mode:
            if self.dropout_fn is None:
                x = F.linear(F.normalize(x_feature, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            else:
                x = F.linear(self.dropout_fn(F.normalize(x_feature, p=2, dim=-1)), F.normalize(self.fc.weight, p=2, dim=-1))

            # x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x_feature)
            # x = self.args.temperature * x
        return x_feature, x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            x_feature, input = self.forward_metric(input)
            return x_feature, input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        global data_imgs
        for batch in dataloader:
            data_imgs, label = [_.cuda() for _ in batch]
            data=self.encode(data_imgs).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'finetune' in self.args.soft_mode:  # further finetune
            print('further finetune?')
            self.update_fc_ft(new_fc,data_imgs,label,session, class_list)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        
    def update_fc_ft(self, new_fc, data_imgs,label,session, class_list=None):
        self.eval()
        optimizer_embedding = torch.optim.SGD(self.parameters(), lr=self.args.lr_new, momentum=0.9)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):


                fc = self.fc.weight[:self.args.base_class + self.args.way * session, :]
                data = self.encode(data_imgs)
                logits = self.get_logits(data, fc)
                # acc = count_acc(logits, label)

                loss = F.cross_entropy(logits, label)
                optimizer_embedding.zero_grad()
                loss.backward()

                optimizer_embedding.step()

# 定义神经网络模型
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(256, 32),
            models.resnet18_swat.resnet18(False)
        )
        self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
        
    def forward(self, x):
        return self.encode(x)

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
