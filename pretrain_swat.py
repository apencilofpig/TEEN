import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
from tqdm import tqdm

# --- 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 定义计算设备 (GPU优先)
print(f"使用设备: {DEVICE}")

# SWaT 数据集特定参数
N_FEATURES = 51 # 特征数量
N_CLASSES = 36 # 类别总数
MAX_SAMPLES_PER_CLASS = 2000 # 每个类别的最大样本数
CSV_FILE_PATH = 'data/swat/data_newlabel.csv' # 请替换为您的CSV文件路径
JSON_ATTACK_PATH = 'data/swat/attack_point.json' # 请替换为您的JSON文件路径

# CVXPY 参数
CVXPY_LAMBDA = 1.0 # 类间项的超参数
CVXPY_C1_NUCLEAR_NORM_LIMIT = 100.0 # 核范数的限制值
CVXPY_SAMPLES_PER_CLASS = 10 # 用于CVXPY的每个类别的样本数 (以保持配对数量可控)
CVXPY_NUM_EPOCHS_W0_UPDATE = 1 # 更新W0并重新求解的次数 (可选, 对于初始化1次通常足够)

# ResNet 参数
RESNET_INITIAL_CHANNELS = 64 # 初始重塑后的通道数 (如果从51进行重塑)
# 对于一维ResNet，W_init后的第一层将进行适配。让W_init的输出(51维)
# 被后续的卷积层重塑或直接使用。我们将W_init的输出(51)
# 作为第一个ResNet块的输入。假设第一个块接收1个通道，长度为51。
RESNET_BLOCK_CHANNELS = [64, 128, 256, 512] # ResNet块中的通道数
RESNET_NUM_BLOCKS = [2, 2, 2, 2] # 每个阶段的残差块数量

# 训练参数
BASE_CLASSES_RATIO = 0.5 # 例如，50%的类别用于基类训练
BATCH_SIZE = 64 # 批量大小
EPOCHS_BASE_TRAINING = 20 # 基类训练轮数 (根据需要调整)
LR_BASE = 0.001 # 基类训练学习率
EPOCHS_INCREMENTAL = 10 # 用于调整分类器的轮数，主干网络冻结

# --- 1. 数据加载和预处理 ---
def load_data(csv_path, json_path):
    print("正在加载数据...")
    # 加载SWaT数据 (假设最后一列是标签，无表头)
    try:
        df = pd.read_csv(csv_path, header=None)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values.astype(int)
    except Exception as e:
        print(f"加载CSV文件出错: {e}")
        print("请确保您的CSV文件有52列 (51个特征 + 1个标签) 并且没有表头。")
        # 如果CSV加载失败，则生成伪数据用于代码流程演示
        print("由于CSV加载失败，正在生成用于演示的伪数据。")
        data = np.random.rand(N_CLASSES * 100, N_FEATURES)
        labels = np.random.randint(0, N_CLASSES, N_CLASSES * 100)
        df = pd.DataFrame(np.concatenate([data, labels.reshape(-1,1)], axis=1))


    # 每个类别最多采样 MAX_SAMPLES_PER_CLASS 个样本
    df_sampled = df.groupby(N_FEATURES, group_keys=False).apply(lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_CLASS)))
    data = df_sampled.iloc[:, :-1].values
    labels = df_sampled.iloc[:, -1].values.astype(int)

    print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")
    print(f"划分前独立标签: {np.unique(labels)}")

    # 特征归一化
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # 加载用于alpha的攻击先验知识
    try:
        with open(json_path, 'r') as f:
            attack_info = json.load(f)
    except Exception as e:
        print(f"加载JSON文件出错: {e}")
        # 如果JSON加载失败，则使用伪攻击信息
        attack_info = [{"attack": i, "points": [j % N_FEATURES]} for i in range(N_CLASSES) for j in range(i % 3 + 1)]


    # 计算alpha (传感器重要性)
    # JSON中的特征索引是0索引的，如果它们对应于列0到N_FEATURES-1
    # 问题描述为 "points: [3]" 等。假设这些是0索引的特征编号。
    alpha_counts = np.zeros(N_FEATURES)
    for item in attack_info:
        # 我们只关心哪些传感器曾被攻击过来计算全局alpha
        for point_idx in item['points']:
            if 0 <= point_idx < N_FEATURES:
                alpha_counts[point_idx] += 1

    if np.sum(alpha_counts) == 0: # 处理alpha计数全为零的情况（例如，attack_info中没有点或所有点都越界）
        print("警告: Alpha计数全为零。使用均匀alpha。")
        alpha_counts = np.ones(N_FEATURES)

    alpha_sensor = alpha_counts / np.sum(alpha_counts) # L1归一化
    alpha_sensor = torch.tensor(alpha_sensor, dtype=torch.float32).to(DEVICE)

    print(f"Alpha (传感器重要性总和): {alpha_sensor.sum().item()}")
    return data, labels, alpha_sensor, scaler


class SWATDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- 2. CVXPY 优化计算初始第一层权重 (W_init) ---
def calculate_W_init_cvxpy(data_np, labels_np, alpha_sensor_np, num_classes_total):
    print("使用 CVXPY 计算 W_init...")
    W_init_cvx = cp.Variable((N_FEATURES, N_FEATURES), name="W_init") # 定义CVXPY优化变量
    alpha_cvx = alpha_sensor_np # 应为numpy数组

    # 为CVXPY使用数据子集以管理复杂性
    selected_indices = []
    for class_id in range(num_classes_total):
        class_indices = np.where(labels_np == class_id)[0]
        if len(class_indices) > 0:
            selected_indices.extend(np.random.choice(class_indices,
                                                     size=min(len(class_indices), CVXPY_SAMPLES_PER_CLASS),
                                                     replace=False))

    if not selected_indices:
        print("警告：没有为CVXPY选择数据。返回随机W_init。")
        return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)

    cvx_data = data_np[selected_indices]
    cvx_labels = labels_np[selected_indices]

    # W_init^0的初始猜测 (用于线性化第二项)
    # 以单位矩阵或随机矩阵开始W0
    W0_np = np.eye(N_FEATURES, dtype=np.float32)
    # W0_np = np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)


    for iter_cvxpy in range(CVXPY_NUM_EPOCHS_W0_UPDATE): # CVXPY迭代（通常1次用于初始化）
        print(f"CVXPY 迭代 {iter_cvxpy + 1}/{CVXPY_NUM_EPOCHS_W0_UPDATE}")
        term1_intra_class_diff = [] # 类内差异项列表
        term2_inter_class_grad_sum = np.zeros_like(W0_np) # 类间差异项的梯度累加

        # 为CVXPY目标函数创建样本对
        # 这可能非常大，需要仔细迭代或采样对
        num_cvx_samples = len(cvx_labels)
        pair_count_intra = 0 # 类内对计数
        pair_count_inter = 0 # 类间对计数

        # 限制对的数量以避免构建CVXPY问题时耗时过长
        MAX_CVX_PAIRS = 2000 # 根据需要调整

        # 类内样本对
        for i in tqdm(range(num_cvx_samples), desc="CVXPY 类内样本对"):
            for j in range(i + 1, num_cvx_samples):
                if cvx_labels[i] == cvx_labels[j]:
                    if pair_count_intra < MAX_CVX_PAIRS:
                        Vij = (cvx_data[i] - cvx_data[j]).reshape(-1, 1) # 列向量
                        # 对于目标函数: alpha^T @ cp.abs(W_init_cvx @ Vij)
                        # cp.abs 逐元素操作。 W_init_cvx @ Vij 是 (N_FEATURES x 1)
                        # alpha_cvx 是 (N_FEATURES,)。所以是 alpha_cvx @ ...
                        term1_intra_class_diff.append(alpha_cvx @ cp.abs(W_init_cvx @ Vij))
                        pair_count_intra +=1
                    else: break
            if pair_count_intra >= MAX_CVX_PAIRS: break

        # 类间样本对 (用于梯度项)
        for i in tqdm(range(num_cvx_samples), desc="CVXPY 类间样本对 (梯度)"):
            for j in range(num_cvx_samples): # 如果对称可以是 i+1，但这里保持通用性
                if cvx_labels[i] != cvx_labels[j]:
                    if pair_count_inter < MAX_CVX_PAIRS:
                        Vij = (cvx_data[i] - cvx_data[j]).reshape(-1, 1) # 列向量
                        W0_Vij = W0_np @ Vij
                        # 梯度项: (alpha_cvx[:, np.newaxis] * np.sign(W0_Vij)) @ Vij.T
                        # alpha_cvx 是 (N_FEATURES,), W0_Vij 是 (N_FEATURES,1)
                        # sgn_W0_Vij 是 (N_FEATURES,1)
                        # alpha_cvx * np.sign(W0_Vij.flatten()) 结果是 (N_FEATURES,)
                        # 然后与 Vij.T 做外积
                        grad_f1_contrib = (alpha_cvx * np.sign(W0_Vij.flatten()))[:, np.newaxis] @ Vij.T
                        term2_inter_class_grad_sum += grad_f1_contrib
                        pair_count_inter +=1
                    else: break
            if pair_count_inter >= MAX_CVX_PAIRS: break

        if not term1_intra_class_diff: # 如果没有找到类内样本对 (例如样本太少)
             print("警告：CVXPY目标函数没有类内样本对。返回随机W_init。")
             return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)

        objective_expr = cp.sum(term1_intra_class_diff) # CVXPY目标函数表达式
        if pair_count_inter > 0 : # 仅当找到类间样本对时才添加该项
            objective_expr -= CVXPY_LAMBDA * cp.trace(term2_inter_class_grad_sum.T @ W_init_cvx)

        constraints = [cp.norm(W_init_cvx, "nuc") <= CVXPY_C1_NUCLEAR_NORM_LIMIT] # CVXPY约束 (核范数约束)
        problem = cp.Problem(cp.Minimize(objective_expr), constraints) # 定义CVXPY问题

        print("正在求解CVXPY问题...")
        try:
            # 如果默认求解器失败，尝试其他求解器
            problem.solve(solver=cp.SCS, verbose=True, max_iters=2500) # SCS适用于大问题
            # problem.solve(solver=cp.ECOS, verbose=True)
        except cp.error.SolverError as e:
            print(f"CVXPY SolverError: {e}。尝试其他求解器或返回随机W_init。")
            try:
                problem.solve(solver=cp.ECOS, verbose=True, max_iters=200) # ECOS适用于中小型问题
            except Exception as e_inner:
                print(f"CVXPY ECOS求解器也失败了: {e_inner}。返回随机W_init。")
                return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)


        if W_init_cvx.value is not None:
            print(f"CVXPY求解完成。状态: {problem.status}")
            W0_np = W_init_cvx.value # 如果 CVXPY_NUM_EPOCHS_W0_UPDATE > 1，则更新W0用于下一次迭代
        else:
            print("CVXPY求解失败或W_init_cvx.value为空。返回随机W_init。")
            return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)

    return W0_np.astype(np.float32) if W0_np is not None else np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)


# --- 3. 一维ResNet模型 ---
class BasicBlock1D(nn.Module): # ResNet的基本块 (1D版本)
    expansion = 1 # 扩展因子
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential() # shortcut连接
        if stride != 1 or in_planes != self.expansion * planes: # 如果维度不匹配，则需要调整shortcut
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 残差连接
        out = torch.relu(out)
        return out

class ResNet1D(nn.Module): # 一维ResNet模型
    def __init__(self, block, num_blocks, num_features_input=N_FEATURES, num_output_initial_linear=N_FEATURES, num_classes_base=10):
        super(ResNet1D, self).__init__()
        self.in_planes = RESNET_BLOCK_CHANNELS[0] # 第一个卷积层（在初始线性层之后）的输入通道数

        # 初始线性层，其权重可能来自CVXPY
        self.initial_linear = nn.Linear(num_features_input, num_output_initial_linear)

        # 重塑: (batch, num_output_initial_linear) -> (batch, 1, num_output_initial_linear)
        # 这意味着第一个ResNet块会将initial_linear的输出视为长度为num_output_initial_linear的单通道序列。
        # _make_layer调用中的第一个卷积层将具有in_planes = 1
        self.conv1_reshape_channel = 1 # 我们将其设置为1，因为我们将 (Batch, 51) 重塑为 (Batch, 1, 51)
                                       # 这意味着ResNet本身的第一个层接收1个输入通道。
                                       # 我们为_make_layer调用调整self.in_planes。

        # ResNet的第一个实际卷积层（在initial_linear和reshape之后）
        # 它会将单通道输入转换为RESNET_BLOCK_CHANNELS[0]个通道
        self.conv1 = nn.Conv1d(self.conv1_reshape_channel, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)

        self.layer1 = self._make_layer(block, RESNET_BLOCK_CHANNELS[0], num_blocks[0], stride=1) # ResNet的第一个阶段
        self.layer2 = self._make_layer(block, RESNET_BLOCK_CHANNELS[1], num_blocks[1], stride=2) # ResNet的第二个阶段
        self.layer3 = self._make_layer(block, RESNET_BLOCK_CHANNELS[2], num_blocks[2], stride=2) # ResNet的第三个阶段
        self.layer4 = self._make_layer(block, RESNET_BLOCK_CHANNELS[3], num_blocks[3], stride=2) # ResNet的第四个阶段
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 自适应平均池化层
        self.fc_classifier = nn.Linear(RESNET_BLOCK_CHANNELS[3] * block.expansion, num_classes_base) # 全连接分类层

    def _make_layer(self, block, planes, num_blocks, stride): # 构建ResNet的一个阶段
        strides = [stride] + [1] * (num_blocks - 1) # 第一个block的步长可能不同
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion # 更新下一层的输入通道数
        return nn.Sequential(*layers)

    def set_initial_linear_weights(self, W_init_tensor): # 设置初始线性层的权重
        if W_init_tensor.shape == self.initial_linear.weight.shape:
            with torch.no_grad():
                self.initial_linear.weight.copy_(W_init_tensor)
                if self.initial_linear.bias is not None:
                    nn.init.zeros_(self.initial_linear.bias)
            print("已成功从CVXPY设置initial_linear层的权重。")
        else:
            print(f"形状不匹配: W_init_tensor {W_init_tensor.shape}, initial_linear.weight {self.initial_linear.weight.shape}。使用默认初始化。")
            # 如果形状不匹配（例如CVXPY失败时），则使用默认初始化
            nn.init.kaiming_normal_(self.initial_linear.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, extract_features=False): # 前向传播
        # x的形状是 (batch, N_FEATURES)
        out = self.initial_linear(x) # (batch, num_output_initial_linear)

        # 为1D卷积重塑: (batch, 1, num_output_initial_linear)
        out = out.unsqueeze(1)

        out = torch.relu(self.bn1(self.conv1(out))) # 重塑后的第一个卷积层

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out) # (batch, channels_final, 1)
        features = out.view(out.size(0), -1) # (batch, channels_final)，展平特征

        if extract_features: # 如果需要提取特征
            return features

        logits = self.fc_classifier(features) # 通过分类器得到logits
        return logits

    def freeze_backbone(self): # 冻结主干网络参数
        for param_name, param in self.named_parameters():
            if 'fc_classifier' not in param_name: # 冻结除分类器外的所有层参数
                param.requires_grad = False
            else: # 确保分类器可训练（如果正在调整）
                 param.requires_grad = True
        print("主干网络已冻结。")

    def unfreeze_backbone(self): # 解冻主干网络参数
        for param in self.parameters():
            param.requires_grad = True
        print("主干网络已解冻。")

    def replace_classifier_with_feature_embedder(self): # 将分类器替换为特征嵌入提取器
        """移除fc_classifier。特征提取将是显式的。"""
        self.fc_classifier = nn.Identity() # 或者实现特定的嵌入逻辑
        print("分类器被替换为恒等映射，用于特征嵌入。")


# --- 4. 训练和评估 ---

def train_epoch(model, dataloader, criterion, optimizer, device): # 训练一个epoch
    model.train() # 设置模型为训练模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in tqdm(dataloader, desc="训练轮次"): # 遍历数据加载器
        inputs, targets = inputs.to(device), targets.to(device) # 数据移至设备

        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs) # 前向传播
        loss = criterion(outputs, targets) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        total_loss += loss.item() * inputs.size(0) # 累加损失
        _, predicted = torch.max(outputs.data, 1) # 获取预测结果
        total_samples += targets.size(0) # 累加样本总数
        correct_predictions += (predicted == targets).sum().item() # 累加正确预测数

    avg_loss = total_loss / total_samples # 计算平均损失
    accuracy = correct_predictions / total_samples # 计算准确率
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device): # 评估模型
    model.eval() # 设置模型为评估模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # 不计算梯度
        for inputs, targets in tqdm(dataloader, desc="评估中"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# --- 主程序逻辑 ---
if __name__ == '__main__':
    # 1. 加载数据和 Alpha
    all_data_np, all_labels_np, alpha_sensor_torch, scaler = load_data(CSV_FILE_PATH, JSON_ATTACK_PATH)

    unique_labels_overall = np.unique(all_labels_np) # 数据集中的所有唯一标签
    print(f"数据集中总的唯一类别数: {len(unique_labels_overall)}")
    if len(unique_labels_overall) == 0:
        raise ValueError("没有加载到数据或没有找到标签。正在退出。")

    # 2. 使用 CVXPY 计算 W_init (第一层权重: 51x51)
    # 确保用于CVXPY的标签覆盖一定范围的类别
    # 如果N_CLASSES很小，这可能是所有类别。
    W_init_np = calculate_W_init_cvxpy(all_data_np, all_labels_np, alpha_sensor_torch.cpu().numpy(), N_CLASSES)
    W_init_torch = torch.tensor(W_init_np, dtype=torch.float32).to(DEVICE)

    # 3. 将数据划分为基类和增量类集合
    # 为简单起见，对标签排序并选择前N个作为基类，其余作为增量类

    n_base_classes = int(N_CLASSES * BASE_CLASSES_RATIO) # 基类数量
    if n_base_classes == 0 and N_CLASSES > 0: n_base_classes = 1 # 如果可能，至少有一个基类
    if n_base_classes == N_CLASSES and N_CLASSES > 1: n_base_classes = N_CLASSES -1 # 至少有一个增量类

    base_class_labels = unique_labels_overall[:n_base_classes] # 基类标签
    incremental_class_labels = unique_labels_overall[n_base_classes:] # 增量类标签

    print(f"基类 ({len(base_class_labels)}): {base_class_labels}")
    print(f"增量类 ({len(incremental_class_labels)}): {incremental_class_labels}")

    if not base_class_labels.size: # 如果没有基类
        print("没有选择基类。跳过基类训练。")
    else:
        base_indices = np.isin(all_labels_np, base_class_labels) # 获取基类数据的索引
        base_data_np, base_labels_np = all_data_np[base_indices], all_labels_np[base_indices]

        # 将基类标签重映射为从0开始的索引，如果它们还不是的话
        # 例如，如果 base_class_labels = [5, 7, 10], 映射为 [0, 1, 2]
        base_label_map = {original_label: new_label for new_label, original_label in enumerate(base_class_labels)}
        base_labels_remapped_np = np.array([base_label_map[lbl] for lbl in base_labels_np])

        # 为基类训练创建DataLoaders
        # 将基类数据进一步划分为训练集/验证集用于基模型训练
        if len(base_data_np) > 1: # 至少需要2个样本才能划分
            train_base_data, val_base_data, train_base_labels, val_base_labels = train_test_split(
                base_data_np, base_labels_remapped_np, test_size=0.2, stratify=base_labels_remapped_np if len(np.unique(base_labels_remapped_np)) > 1 else None
            )
            train_base_dataset = SWATDataset(train_base_data, train_base_labels)
            val_base_dataset = SWATDataset(val_base_data, val_base_labels)
            train_base_loader = DataLoader(train_base_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_base_loader = DataLoader(val_base_dataset, batch_size=BATCH_SIZE, shuffle=False)
        elif len(base_data_np) == 1:
            print("基类数据中只有一个样本。用它进行训练，无验证。")
            train_base_dataset = SWATDataset(base_data_np, base_labels_remapped_np)
            train_base_loader = DataLoader(train_base_dataset, batch_size=1, shuffle=True)
            val_base_loader = None # 无验证
        else: # 没有基类数据
            print("没有用于基类训练的数据。")
            train_base_loader = None
            val_base_loader = None


        # 4. 初始化和训练基模型
        if train_base_loader: # 确保有基类数据加载器
            print("\n--- 基模型训练 ---")
            # ResNet的initial_linear层是 51 -> 51。
            # 其fc_classifier将输出num_base_classes_remapped个logit。
            num_base_classes_remapped = len(np.unique(base_labels_remapped_np)) # 重映射后的基类数量

            base_model = ResNet1D(BasicBlock1D, RESNET_NUM_BLOCKS, num_classes_base=num_base_classes_remapped).to(DEVICE)
            base_model.set_initial_linear_weights(W_init_torch) # 初始化第一个线性层

            # 初始化其他层 (卷积层使用Kaiming初始化等) - PyTorch对大多数层默认执行此操作。

            criterion_base = nn.CrossEntropyLoss() # 基类训练损失函数
            optimizer_base = optim.Adam(base_model.parameters(), lr=LR_BASE) # 基类训练优化器

            for epoch in range(EPOCHS_BASE_TRAINING): # 训练循环
                train_loss, train_acc = train_epoch(base_model, train_base_loader, criterion_base, optimizer_base, DEVICE)
                print(f"基类 Epoch {epoch+1}/{EPOCHS_BASE_TRAINING} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
                if val_base_loader:
                    val_loss, val_acc = evaluate(base_model, val_base_loader, criterion_base, DEVICE)
                    print(f"基类 Epoch {epoch+1}/{EPOCHS_BASE_TRAINING} - 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        else:
            print("由于没有基类数据，跳过基模型训练。")
            base_model = None # 如果没有基类数据，则没有基模型

    # 5. 增量学习阶段
    if base_model and incremental_class_labels.size > 0: # 确保基模型已训练且有增量类
        print("\n--- 增量学习阶段 ---")
        base_model.freeze_backbone() # 冻结主干网络
        base_model.replace_classifier_with_feature_embedder() # 现在 model(x, extract_features=True) 是隐式的

        # 准备增量数据
        incremental_indices = np.isin(all_labels_np, incremental_class_labels) # 获取增量数据的索引
        incremental_data_np, incremental_labels_np = all_data_np[incremental_indices], all_labels_np[incremental_indices]

        if len(incremental_data_np) == 0: # 如果没有增量数据
            print("没有用于增量学习的数据。跳过。")
        else:
            # 存储所有已知类别（基类 + 增量类）的原型（平均嵌入）
            all_prototypes = {} # 存储为 {全局重映射标签: 原型张量}

            # 为增量学习后所有已知类别创建全局标签映射
            # 例如, 如果 unique_labels_overall = [0,1,2,5,7]
            # base_class_labels = [0,1,2] -> 重映射为 [0,1,2]
            # incremental_class_labels = [5,7] -> 对于组合分类器重映射为 [3,4]
            known_class_labels_ordered = np.concatenate([base_class_labels, incremental_class_labels])
            global_label_map = {original_label: new_label for new_label, original_label in enumerate(known_class_labels_ordered)}
            num_total_known_classes = len(known_class_labels_ordered) # 已知类别总数

            print("正在为基类计算原型...")
            if len(base_data_np) > 0: # 确保 base_data_np 已填充
                base_data_torch = torch.tensor(base_data_np, dtype=torch.float32).to(DEVICE)
                base_labels_original_torch = torch.tensor(base_labels_np, dtype=torch.long).to(DEVICE)

                with torch.no_grad(): # 提取基类特征
                    base_features = base_model(base_data_torch, extract_features=True)

                for original_base_label in base_class_labels: # 计算每个基类的原型
                    mask = (base_labels_original_torch == original_base_label)
                    if mask.sum() > 0:
                        prototype = base_features[mask].mean(dim=0)
                        global_remapped_label = global_label_map[original_base_label]
                        all_prototypes[global_remapped_label] = prototype

            print("正在为增量类计算原型...")
            # 对于增量类的“训练”，我们只计算它们的原型
            # 在一个真实的小样本场景中，每个增量类只会使用非常少的样本
            incremental_data_torch = torch.tensor(incremental_data_np, dtype=torch.float32).to(DEVICE)
            incremental_labels_original_torch = torch.tensor(incremental_labels_np, dtype=torch.long).to(DEVICE)

            with torch.no_grad(): # 提取增量类特征
                incremental_features = base_model(incremental_data_torch, extract_features=True)

            for original_incr_label in incremental_class_labels: # 计算每个增量类的原型
                mask = (incremental_labels_original_torch == original_incr_label)
                if mask.sum() > 0:
                    prototype = incremental_features[mask].mean(dim=0)
                    global_remapped_label = global_label_map[original_incr_label]
                    all_prototypes[global_remapped_label] = prototype

            print(f"计算得到的原型总数: {len(all_prototypes)}")

            # 在增量数据（或包含混合类的独立测试集）上进行评估
            # 使用与原型的余弦相似度
            if len(all_prototypes) > 0 and len(incremental_data_np) > 0:
                print("使用原型在增量数据上进行评估...")
                correct_incr = 0
                total_incr = 0

                # 将原型堆叠成张量以便高效计算余弦相似度
                # 确保它们按global_remapped_label排序
                prototype_tensor_list = []
                for i in range(num_total_known_classes):
                    if i in all_prototypes:
                        prototype_tensor_list.append(all_prototypes[i])
                    else: # 如果所有类别都有样本，则不应发生
                        prototype_tensor_list.append(torch.zeros_like(next(iter(all_prototypes.values())))) # 占位符

                prototype_matrix = torch.stack(prototype_tensor_list).to(DEVICE) # (已知类别总数, 特征维度)

                # 使用先前提取的incremental_features进行评估
                # 或者根据需要重新提取，尤其是在保留测试集上评估时
                query_features = incremental_features # (增量样本数, 特征维度)

                # 余弦相似度: (A @ B.T) / (||A|| * ||B||)
                query_features_norm = query_features / query_features.norm(dim=1, keepdim=True)
                prototype_matrix_norm = prototype_matrix / prototype_matrix.norm(dim=1, keepdim=True)

                similarities = query_features_norm @ prototype_matrix_norm.T # (增量样本数, 已知类别总数)
                predicted_global_remapped_labels = torch.argmax(similarities, dim=1) # 预测的全局重映射标签

                # 将原始增量标签转换为其全局重映射版本以进行比较
                true_global_remapped_labels = torch.tensor(
                    [global_label_map[orig_label] for orig_label in incremental_labels_np],
                    dtype=torch.long
                ).to(DEVICE)

                correct_incr = (predicted_global_remapped_labels == true_global_remapped_labels).sum().item() # 正确预测数
                total_incr = len(incremental_labels_np) # 增量样本总数

                if total_incr > 0:
                    incr_accuracy = correct_incr / total_incr # 增量任务准确率
                    print(f"增量任务准确率 (在增量数据上): {incr_accuracy:.4f} ({correct_incr}/{total_incr})")
                else:
                    print("没有增量样本可供评估。")
            else:
                print("跳过增量评估：没有原型或没有增量数据。")
    elif not base_model:
        print("未训练基模型，跳过增量学习。")
    elif not incremental_class_labels.size: # 如果没有定义增量类
        print("没有定义增量类别，跳过增量学习阶段。")

    print("\n--- 脚本执行完毕 ---")