import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)

###############################################
# 1. 数据生成与预处理
###############################################
# 模拟生成合成数据：特征维度 41
input_dim = 41
num_old_classes = 11  # 旧类别数
num_new_classes = 2   # 新类别数

def generate_synthetic_data(num_samples, num_classes):
    """
    生成合成数据，每个类别正态分布产生数据
    """
    X = []
    y = []
    for c in range(num_classes):
        # 每个类别中心随机生成
        center = np.random.randn(input_dim) * 5
        # 生成对应类别数据，方差较小
        X.append(center + np.random.randn(num_samples, input_dim))
        y.append(np.full(num_samples, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    return X.astype(np.float32), y.astype(np.int64)

# 为旧类别生成数据（数量较大，用于预训练旧模型）
X_old, y_old = generate_synthetic_data(300, num_old_classes)
# 为新类别生成数据（few-shot设置，每类仅有 10 个样本）
X_new, y_new = generate_synthetic_data(10, num_new_classes)

# 拆分测试集（旧类别）
X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(
    X_old, y_old, test_size=0.2, random_state=42)

###############################################
# 2. 定义网络结构
###############################################
class FeatureExtractor(nn.Module):
    """
    简单的全连接网络作为特征提取器
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ClassifierHead(nn.Module):
    """
    分类头，输出 logits
    """
    def __init__(self, feature_dim, num_classes):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# 整体模型由特征提取器和分类头组成
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.classifier = ClassifierHead(hidden_dim, num_classes)
    
    def forward(self, x):
        feat = self.feature_extractor(x)
        logits = self.classifier(feat)
        return logits

###############################################
# 3. 预训练旧类别检测模型（fold）
###############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
old_model = SimpleClassifier(input_dim=input_dim, hidden_dim=64, num_classes=num_old_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_old = optim.Adam(old_model.parameters(), lr=0.001)

# 进行简单训练
num_epochs = 20
X_old_train_tensor = torch.from_numpy(X_old_train).to(device)
y_old_train_tensor = torch.from_numpy(y_old_train).to(device)

print("预训练旧类别模型 ...")
for epoch in range(num_epochs):
    optimizer_old.zero_grad()
    logits = old_model(X_old_train_tensor)
    loss = criterion(logits, y_old_train_tensor)
    loss.backward()
    optimizer_old.step()
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# 冻结旧模型参数
for param in old_model.parameters():
    param.requires_grad = False

###############################################
# 4. 初始训练阶段：使用元学习训练新类别检测模型（fnew）
###############################################
# fnew 模型：用于分类新类别。这里我们让 fnew 与旧类别模型结构相同，但分类头的输出类别为新类别数。
fnew = SimpleClassifier(input_dim=input_dim, hidden_dim=64, num_classes=num_new_classes).to(device)
meta_optimizer = optim.Adam(fnew.parameters(), lr=0.01)

# 参数设置：K-shot 元学习
K = 5  # 支持集每类样本数
Q = 5  # 查询集每类样本数
meta_batch_size = 4  # 每次 meta-update 中选择的任务数
inner_steps = 5  # 内部循环更新次数
inner_lr = 0.01
meta_iterations = 100

# 为便于构造元任务，我们将旧类别训练数据用于元训练（算法1、2）。
# 将旧类别数据按类别存为字典
def create_class_dict(X, y):
    class_dict = {}
    for xi, yi in zip(X, y):
        class_dict.setdefault(yi, []).append(xi)
    # 转换为 numpy 数组
    for yi in class_dict:
        class_dict[yi] = np.stack(class_dict[yi], axis=0)
    return class_dict

old_class_dict = create_class_dict(X_old_train, y_old_train)

def sample_meta_task(class_dict, num_classes, K, Q):
    """
    从旧类别数据中随机采样一个 meta-task：随机选择 num_classes 类，
    每类采样 K+Q 个样本，分别作为支持集和查询集。
    返回支持集和查询集及对应标签（从 0 到 num_classes-1）
    """
    selected_classes = np.random.choice(list(class_dict.keys()), num_classes, replace=False)
    support_x = []
    support_y = []
    query_x = []
    query_y = []
    for new_label, c in enumerate(selected_classes):
        samples = class_dict[c]
        indices = np.random.choice(len(samples), K+Q, replace=False)
        support_samples = samples[indices[:K]]
        query_samples = samples[indices[K:]]
        support_x.append(support_samples)
        support_y.append(np.full(K, new_label))
        query_x.append(query_samples)
        query_y.append(np.full(Q, new_label))
    support_x = np.concatenate(support_x, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    query_x = np.concatenate(query_x, axis=0)
    query_y = np.concatenate(query_y, axis=0)
    return support_x, support_y, query_x, query_y

# 使用第一阶近似实现 MAML 算法（算法2）
print("\n元学习阶段训练新类别检测模型 ...")
for it in range(meta_iterations):
    meta_loss = 0.0
    meta_optimizer.zero_grad()
    # 每个 meta-batch 内采样多个任务，任务类别数设为 num_meta_classes
    num_meta_classes = num_new_classes  # 为简单起见，此处任务类别数与新类别数一致
    for _ in range(meta_batch_size):
        # 采样一个 meta-task，支持集和查询集的标签重置为 0,1,...,num_meta_classes-1
        support_x, support_y, query_x, query_y = sample_meta_task(old_class_dict, num_meta_classes, K, Q)
        # 将数据转为 tensor，并放在 device 上
        support_x = torch.from_numpy(support_x).to(device)
        support_y = torch.from_numpy(support_y).to(device)
        query_x = torch.from_numpy(query_x).to(device)
        query_y = torch.from_numpy(query_y).to(device)
        
        # 保存初始 fnew 参数
        fast_weights = {name: param.clone() for name, param in fnew.named_parameters()}
        
        # 内循环：对支持集进行若干次更新（内循环梯度）
        for _ in range(inner_steps):
            logits = fnew(support_x)
            loss = criterion(logits, support_y)
            grads = torch.autograd.grad(loss, fnew.parameters(), create_graph=True)
            # 手动更新参数（一步梯度下降）
            fast_weights = {name: param - inner_lr * grad
                            for ((name, param), grad) in zip(fnew.named_parameters(), grads)}
            # 将更新后的参数载入模型（采用参数替换方式）
            def forward_with_weights(model, x, weights):
                # 使用当前权重对模型进行前向运算
                x = F.relu(F.linear(x, weights['feature_extractor.fc1.weight'], weights['feature_extractor.fc1.bias']))
                x = F.relu(F.linear(x, weights['feature_extractor.fc2.weight'], weights['feature_extractor.fc2.bias']))
                logits = F.linear(x, weights['classifier.fc.weight'], weights['classifier.fc.bias'])
                return logits
            logits = forward_with_weights(fnew, support_x, fast_weights)
            loss = criterion(logits, support_y)
        
        # 在查询集上计算损失
        logits_q = forward_with_weights(fnew, query_x, fast_weights)
        task_loss = criterion(logits_q, query_y)
        meta_loss += task_loss
    # 求平均 loss
    meta_loss = meta_loss / meta_batch_size
    meta_loss.backward()
    meta_optimizer.step()
    if (it+1) % 20 == 0:
        print(f"Meta-iteration {it+1}/{meta_iterations}, Meta Loss: {meta_loss.item():.4f}")

###############################################
# 5. 增量训练阶段：模型融合（Model Merging）
###############################################
# 假设此时已有旧类别模型 old_model 和新类别模型 fnew（参数经元学习得到）
# 下一步：利用少量新样本以及旧类部分样本，训练一层简单全连接网络，将两个模型输出的 logits 拼接后作为融合特征。

class MergeClassifier(nn.Module):
    """
    三层全连接网络，用于将旧类和新类模型的 logits 融合后分类
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MergeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 为了融合，我们假设 old_model 给出旧类 logits (维度 num_old_classes)
# 而 fnew 给出新类 logits (维度 num_new_classes)
# 融合后最后分类类别数为 (num_old_classes + num_new_classes)
final_num_classes = num_old_classes + num_new_classes
# 融合输入维度为旧模型 logits 数 + 新模型 logits 数
merge_input_dim = num_old_classes + num_new_classes
merge_model = MergeClassifier(input_dim=merge_input_dim, hidden_dim=64, num_classes=final_num_classes).to(device)

# 构造增量训练数据：将新样本和部分旧类样本拼接在一起
# 这里为了示例，我们从旧训练集中随机抽取一部分数据作为增量训练时旧类数据
X_old_inc, _, y_old_inc, _ = train_test_split(X_old_train, y_old_train, test_size=0.9, random_state=42)
# 调整新类别标签，使新类别标签在增量训练中从 num_old_classes 开始
y_new_inc = y_new + num_old_classes

# 合并数据
X_inc = np.concatenate([X_old_inc, X_new], axis=0)
y_inc = np.concatenate([y_old_inc, y_new_inc], axis=0)

# 划分支持集和测试集（本例中直接随机划分）
X_inc_train, X_inc_test, y_inc_train, y_inc_test = train_test_split(
    X_inc, y_inc, test_size=0.5, random_state=42)

# 转为 tensor
X_inc_train_tensor = torch.from_numpy(X_inc_train).to(device)
y_inc_train_tensor = torch.from_numpy(y_inc_train).to(device)
X_inc_test_tensor = torch.from_numpy(X_inc_test).to(device)
y_inc_test_tensor = torch.from_numpy(y_inc_test).to(device)

# 定义优化器
optimizer_merge = optim.Adam(list(merge_model.parameters()) + list(fnew.parameters()), lr=0.01)
merge_epochs = 50

print("\n增量训练阶段（模型融合） ...")
for epoch in range(merge_epochs):
    optimizer_merge.zero_grad()
    # 对每个样本，获取旧模型与新模型的 logits
    with torch.no_grad():
        logits_old = old_model(X_inc_train_tensor)  # 尺寸：[N, num_old_classes]
    logits_new = fnew(X_inc_train_tensor)  # 尺寸：[N, num_new_classes]
    # 拼接 logits
    merge_input = torch.cat([logits_old, logits_new], dim=1)
    logits_final = merge_model(merge_input)
    loss_merge = criterion(logits_final, y_inc_train_tensor)
    loss_merge.backward()
    optimizer_merge.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{merge_epochs}, Loss: {loss_merge.item():.4f}")

###############################################
# 6. 最终测试
###############################################
with torch.no_grad():
    logits_old_test = old_model(X_inc_test_tensor)
    logits_new_test = fnew(X_inc_test_tensor)
    merge_input_test = torch.cat([logits_old_test, logits_new_test], dim=1)
    logits_final_test = merge_model(merge_input_test)
    pred = torch.argmax(logits_final_test, dim=1)
    acc = (pred == y_inc_test_tensor).float().mean().item()
    print(f"\n最终融合模型在增量测试集上的准确率: {acc*100:.2f}%")
