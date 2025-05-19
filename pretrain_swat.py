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
N_CLASSES = 36 # 数据集中预期的类别总数 (26 base + 10 incremental)
MAX_SAMPLES_PER_CLASS = 2000 # 每个类别的最大样本数
CSV_FILE_PATH = 'data/swat/data_newlabel.csv' # 请替换为您的CSV文件路径
JSON_ATTACK_PATH = 'data/swat/attack_point.json' # 请替换为您的JSON文件路径

# CVXPY 参数
CVXPY_LAMBDA = 1.0 # 类间项的超参数
CVXPY_C1_NUCLEAR_NORM_LIMIT = 100.0 # 核范数的限制值
CVXPY_SAMPLES_PER_CLASS = 10 # 用于CVXPY的每个类别的样本数 (以保持配对数量可控)
CVXPY_NUM_EPOCHS_W0_UPDATE = 1 # 更新W0并重新求解的次数 (可选, 对于初始化1次通常足够)

# ResNet 参数
RESNET_BLOCK_CHANNELS = [64, 128, 256, 512] # ResNet块中的通道数
RESNET_NUM_BLOCKS = [2, 2, 2, 2] # 每个阶段的残差块数量

# 训练参数
# BASE_CLASSES_RATIO 被 N_BASE_CLASSES 取代
N_BASE_CLASSES = 16 # 基类数量
N_INCREMENTAL_STAGES = 10 # 增量阶段数量
N_INCREMENTAL_CLASSES_PER_STAGE = 2 # 每个增量阶段学习的类别数量

BATCH_SIZE = 64 # 批量大小
EPOCHS_BASE_TRAINING = 20 # 基类训练轮数 (根据需要调整)
LR_BASE = 0.001 # 基类训练学习率
# EPOCHS_INCREMENTAL 不再需要，因为增量学习是基于原型计算

# --- 1. 数据加载和预处理 ---
def load_data(csv_path, json_path):
    print("正在加载数据...")
    try:
        df = pd.read_csv(csv_path, header=None)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values.astype(int)
    except Exception as e:
        print(f"加载CSV文件出错: {e}")
        print("请确保您的CSV文件有52列 (51个特征 + 1个标签) 并且没有表头。")
        print("由于CSV加载失败，正在生成用于演示的伪数据。")
        data = np.random.rand(N_CLASSES * 100, N_FEATURES)
        labels = np.random.randint(0, N_CLASSES, N_CLASSES * 100) # 确保伪标签在0到N_CLASSES-1范围内
        df = pd.DataFrame(np.concatenate([data, labels.reshape(-1,1)], axis=1))
        df.columns = list(range(N_FEATURES)) + ['label'] # 为伪数据添加列名以便groupby

    # 按标签列（最后一列）分组并采样
    label_column_index = N_FEATURES # 标签列的索引 (0-indexed)
    df_sampled = df.groupby(label_column_index, group_keys=False).apply(lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_CLASS)))
    data = df_sampled.iloc[:, :N_FEATURES].values
    labels = df_sampled.iloc[:, label_column_index].values.astype(int)

    print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")
    print(f"划分前独立标签: {np.unique(labels)}")

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    try:
        with open(json_path, 'r') as f:
            attack_info = json.load(f)
    except Exception as e:
        print(f"加载JSON文件出错: {e}")
        attack_info = [{"attack": i, "points": [j % N_FEATURES]} for i in range(N_CLASSES) for j in range(i % 3 + 1)]

    alpha_counts = np.zeros(N_FEATURES)
    for item in attack_info:
        for point_idx in item['points']:
            if 0 <= point_idx < N_FEATURES:
                alpha_counts[point_idx] += 1
    if np.sum(alpha_counts) == 0:
        print("警告: Alpha计数全为零。使用均匀alpha。")
        alpha_counts = np.ones(N_FEATURES)
    alpha_sensor = alpha_counts / np.sum(alpha_counts)
    alpha_sensor = torch.tensor(alpha_sensor, dtype=torch.float32).to(DEVICE)
    print(f"Alpha (传感器重要性总和): {alpha_sensor.sum().item()}")
    return data, labels, alpha_sensor, scaler

class SWATDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# --- 2. CVXPY 优化 ---
def calculate_W_init_cvxpy(data_np, labels_np, alpha_sensor_np, num_unique_classes_for_cvxpy):
    print("使用 CVXPY 计算 W_init...")
    W_init_cvx = cp.Variable((N_FEATURES, N_FEATURES), name="W_init")
    alpha_cvx = alpha_sensor_np
    selected_indices = []
    
    # CVXPY阶段使用数据中实际存在的唯一标签
    unique_labels_in_data = np.unique(labels_np)

    for class_id_orig in unique_labels_in_data: # 遍历原始标签
        class_indices = np.where(labels_np == class_id_orig)[0]
        if len(class_indices) > 0:
            selected_indices.extend(np.random.choice(class_indices,
                                                     size=min(len(class_indices), CVXPY_SAMPLES_PER_CLASS),
                                                     replace=False))
    if not selected_indices:
        print("警告：没有为CVXPY选择数据。返回随机W_init。")
        return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)

    cvx_data = data_np[selected_indices]
    cvx_labels = labels_np[selected_indices]
    W0_np = np.eye(N_FEATURES, dtype=np.float32)

    for iter_cvxpy in range(CVXPY_NUM_EPOCHS_W0_UPDATE):
        print(f"CVXPY 迭代 {iter_cvxpy + 1}/{CVXPY_NUM_EPOCHS_W0_UPDATE}")
        term1_intra_class_diff = []
        term2_inter_class_grad_sum = np.zeros_like(W0_np)
        num_cvx_samples = len(cvx_labels)
        pair_count_intra, pair_count_inter = 0, 0
        MAX_CVX_PAIRS = 2000

        for i in tqdm(range(num_cvx_samples), desc="CVXPY 类内样本对", leave=False):
            for j in range(i + 1, num_cvx_samples):
                if cvx_labels[i] == cvx_labels[j]:
                    if pair_count_intra < MAX_CVX_PAIRS:
                        Vij = (cvx_data[i] - cvx_data[j]).reshape(-1, 1)
                        term1_intra_class_diff.append(alpha_cvx @ cp.abs(W_init_cvx @ Vij))
                        pair_count_intra +=1
                    else: break
            if pair_count_intra >= MAX_CVX_PAIRS: break
        for i in tqdm(range(num_cvx_samples), desc="CVXPY 类间样本对 (梯度)", leave=False):
            for j in range(num_cvx_samples):
                if cvx_labels[i] != cvx_labels[j]:
                    if pair_count_inter < MAX_CVX_PAIRS:
                        Vij = (cvx_data[i] - cvx_data[j]).reshape(-1, 1)
                        W0_Vij = W0_np @ Vij
                        grad_f1_contrib = (alpha_cvx * np.sign(W0_Vij.flatten()))[:, np.newaxis] @ Vij.T
                        term2_inter_class_grad_sum += grad_f1_contrib
                        pair_count_inter +=1
                    else: break
            if pair_count_inter >= MAX_CVX_PAIRS: break
        if not term1_intra_class_diff:
             print("警告：CVXPY目标函数没有类内样本对。返回随机W_init。")
             return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)
        objective_expr = cp.sum(term1_intra_class_diff)
        if pair_count_inter > 0 :
            objective_expr -= CVXPY_LAMBDA * cp.trace(term2_inter_class_grad_sum.T @ W_init_cvx)
        constraints = [cp.norm(W_init_cvx, "nuc") <= CVXPY_C1_NUCLEAR_NORM_LIMIT]
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        print("正在求解CVXPY问题...")
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=2500) # 减少冗余输出
        except cp.error.SolverError as e:
            print(f"CVXPY SolverError: {e}。尝试ECOS。")
            try: problem.solve(solver=cp.ECOS, verbose=False, max_iters=200)
            except Exception as e_inner:
                print(f"CVXPY ECOS求解器也失败了: {e_inner}。返回随机W_init。")
                return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)
        if W_init_cvx.value is not None:
            print(f"CVXPY求解完成。状态: {problem.status}")
            W0_np = W_init_cvx.value
        else:
            print("CVXPY求解失败或W_init_cvx.value为空。返回随机W_init。")
            return np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)
    return W0_np.astype(np.float32) if W0_np is not None else np.random.randn(N_FEATURES, N_FEATURES).astype(np.float32)

# --- 3. 一维ResNet模型 ---
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_features_input=N_FEATURES, num_output_initial_linear=N_FEATURES, num_classes_output=10): # num_classes_output for fc_classifier
        super(ResNet1D, self).__init__()
        self.in_planes = RESNET_BLOCK_CHANNELS[0]
        self.initial_linear = nn.Linear(num_features_input, num_output_initial_linear)
        self.conv1_reshape_channel = 1
        self.conv1 = nn.Conv1d(self.conv1_reshape_channel, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_planes)
        self.layer1 = self._make_layer(block, RESNET_BLOCK_CHANNELS[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, RESNET_BLOCK_CHANNELS[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, RESNET_BLOCK_CHANNELS[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, RESNET_BLOCK_CHANNELS[3], num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_classifier = nn.Linear(RESNET_BLOCK_CHANNELS[3] * block.expansion, num_classes_output) # 用于基类训练

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_initial_linear_weights(self, W_init_tensor):
        if W_init_tensor.shape == self.initial_linear.weight.shape:
            with torch.no_grad(): self.initial_linear.weight.copy_(W_init_tensor)
            if self.initial_linear.bias is not None: nn.init.zeros_(self.initial_linear.bias)
            print("已成功从CVXPY设置initial_linear层的权重。")
        else:
            print(f"形状不匹配: W_init_tensor {W_init_tensor.shape}, initial_linear.weight {self.initial_linear.weight.shape}。使用默认初始化。")
            nn.init.kaiming_normal_(self.initial_linear.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, extract_features=False):
        out = self.initial_linear(x)
        out = out.unsqueeze(1)
        out = torch.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        if extract_features: return features
        logits = self.fc_classifier(features)
        return logits

    def freeze_backbone(self):
        for param_name, param in self.named_parameters():
            if 'fc_classifier' not in param_name: param.requires_grad = False
            else: param.requires_grad = True
        print("主干网络已冻结。")

    def replace_fc_classifier_for_prototypes(self): # 重命名以明确用途
        self.fc_classifier = nn.Identity() # 在原型方法中，我们不需要原始分类器
        print("fc_classifier已替换为Identity，用于原型特征提取。")

# --- 4. 训练和评估 ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    for inputs, targets in tqdm(dataloader, desc="训练轮次", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, targets)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0); correct_predictions += (predicted == targets).sum().item()
    return total_loss / total_samples, correct_predictions / total_samples

def evaluate_with_classifier(model, dataloader, criterion, device): # 用于基类训练评估
    model.eval()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="评估(分类器)", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs); loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0); correct_predictions += (predicted == targets).sum().item()
    return total_loss / total_samples, correct_predictions / total_samples

def evaluate_with_prototypes(model, test_data_tensor, test_labels_remapped_tensor, prototypes_dict, num_known_classes_remapped, device):
    model.eval()
    correct_preds = 0
    total_samples_test = test_labels_remapped_tensor.size(0)

    if total_samples_test == 0 or not prototypes_dict: return 0.0
    
    # 为测试数据分批提取特征以避免OOM
    features_list = []
    test_dataloader_eval = DataLoader(TensorDataset(test_data_tensor), batch_size=BATCH_SIZE*2, shuffle=False) # 使用更大的batch
    with torch.no_grad():
        for (batch_data,) in tqdm(test_dataloader_eval, desc="提取测试特征(原型评估)", leave=False):
             features_list.append(model(batch_data.to(device), extract_features=True))
    query_features = torch.cat(features_list, dim=0)


    feature_dim = next(iter(prototypes_dict.values())).shape[0]
    prototype_tensor_list = []
    for i in range(num_known_classes_remapped): # 使用重映射后的已知类别总数
        if i in prototypes_dict: prototype_tensor_list.append(prototypes_dict[i])
        else:
            print(f"警告: 类别 {i} (全局重映射后) 的原型缺失。使用零向量。")
            prototype_tensor_list.append(torch.zeros(feature_dim, device=device))
    if not prototype_tensor_list: return 0.0
    prototype_matrix = torch.stack(prototype_tensor_list).to(device)

    query_features_norm = query_features / (query_features.norm(dim=1, keepdim=True) + 1e-8)
    prototype_matrix_norm = prototype_matrix / (prototype_matrix.norm(dim=1, keepdim=True) + 1e-8)
    similarities = query_features_norm @ prototype_matrix_norm.T
    predicted_global_remapped_labels = torch.argmax(similarities, dim=1)
    correct_preds = (predicted_global_remapped_labels == test_labels_remapped_tensor.to(device)).sum().item() # 确保标签在同一设备
    return correct_preds / total_samples_test if total_samples_test > 0 else 0.0

# --- 主程序逻辑 ---
if __name__ == '__main__':
    all_data_np, all_labels_np, alpha_sensor_torch, scaler = load_data(CSV_FILE_PATH, JSON_ATTACK_PATH)
    unique_labels_overall = np.unique(all_labels_np)
    print(f"数据集中总的唯一类别数: {len(unique_labels_overall)}")

    if len(unique_labels_overall) < N_CLASSES:
        print(f"警告：数据集中只有 {len(unique_labels_overall)} 个唯一标签，但期望有 {N_CLASSES} 个。将使用实际存在的标签数。")
        # N_CLASSES = len(unique_labels_overall) # 可以选择调整N_CLASSES，但后续划分可能需要检查
    
    # 确保我们不会请求比可用类别更多的类别
    actual_n_base_classes = min(N_BASE_CLASSES, len(unique_labels_overall))
    if actual_n_base_classes < N_BASE_CLASSES:
        print(f"警告: 请求 {N_BASE_CLASSES} 个基类, 但只有 {actual_n_base_classes} 个可用。")

    base_class_labels = unique_labels_overall[:actual_n_base_classes]
    remaining_labels_for_incremental = unique_labels_overall[actual_n_base_classes:]
    
    incremental_stages_orig_labels = [] # 存储每个增量阶段的原始标签
    for i in range(N_INCREMENTAL_STAGES):
        start_idx = i * N_INCREMENTAL_CLASSES_PER_STAGE
        end_idx = start_idx + N_INCREMENTAL_CLASSES_PER_STAGE
        stage_labels = remaining_labels_for_incremental[start_idx:end_idx]
        if len(stage_labels) > 0: incremental_stages_orig_labels.append(stage_labels)
        else: break # 如果没有足够的标签用于当前阶段，则停止
    
    actual_n_incremental_stages = len(incremental_stages_orig_labels)
    if actual_n_incremental_stages < N_INCREMENTAL_STAGES:
        print(f"警告: 请求 {N_INCREMENTAL_STAGES} 个增量阶段, 但只有足够的标签形成 {actual_n_incremental_stages} 个阶段。")

    print(f"基类 ({len(base_class_labels)}): {base_class_labels if len(base_class_labels) > 0 else '无'}")
    for i, stage_lbls in enumerate(incremental_stages_orig_labels):
        print(f"增量阶段 {i+1} 类别 ({len(stage_lbls)}): {stage_lbls}")

    # CVXPY 初始化 (仅执行一次)
    W_init_np = calculate_W_init_cvxpy(all_data_np, all_labels_np, alpha_sensor_torch.cpu().numpy(), len(unique_labels_overall))
    W_init_torch = torch.tensor(W_init_np, dtype=torch.float32).to(DEVICE)

    accuracies_over_stages = [] # 存储 acc0, acc1, acc2

    # --- 基类训练阶段 ---
    print("\n--- 基类训练阶段 ---")
    if len(base_class_labels) > 0:
        base_indices = np.isin(all_labels_np, base_class_labels)
        base_data_np_train_val, base_labels_np_train_val_orig = all_data_np[base_indices], all_labels_np[base_indices]

        # 基类标签重映射 (0 到 N_BASE_CLASSES-1)
        base_label_map_func = {orig_label: i for i, orig_label in enumerate(base_class_labels)}
        base_labels_np_train_val_remapped = np.array([base_label_map_func[lbl] for lbl in base_labels_np_train_val_orig])
        
        num_actual_base_classes_remapped = len(np.unique(base_labels_np_train_val_remapped))

        train_base_data, val_base_data, train_base_labels_remapped, val_base_labels_remapped = train_test_split(
            base_data_np_train_val, base_labels_np_train_val_remapped, test_size=0.2, 
            stratify=base_labels_np_train_val_remapped if num_actual_base_classes_remapped > 1 else None, random_state=42
        )
        train_base_loader = DataLoader(SWATDataset(train_base_data, train_base_labels_remapped), batch_size=BATCH_SIZE, shuffle=True)
        val_base_loader = DataLoader(SWATDataset(val_base_data, val_base_labels_remapped), batch_size=BATCH_SIZE, shuffle=False)

        base_model = ResNet1D(BasicBlock1D, RESNET_NUM_BLOCKS, num_classes_output=num_actual_base_classes_remapped).to(DEVICE)
        base_model.set_initial_linear_weights(W_init_torch)
        criterion_base = nn.CrossEntropyLoss()
        optimizer_base = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=LR_BASE) #确保只优化可训练参数

        for epoch in range(EPOCHS_BASE_TRAINING):
            train_loss, train_acc = train_epoch(base_model, train_base_loader, criterion_base, optimizer_base, DEVICE)
            print(f"基类 Epoch {epoch+1}/{EPOCHS_BASE_TRAINING} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 使用分类器在验证集上评估acc0
        val_loss_base, acc0 = evaluate_with_classifier(base_model, val_base_loader, criterion_base, DEVICE)
        print(f"基类阶段验证准确率 (acc0): {acc0:.4f}")
        accuracies_over_stages.append(acc0)
        
        # 基类训练完成后，准备模型用于原型方法
        base_model.freeze_backbone()
        base_model.replace_fc_classifier_for_prototypes()
    else:
        print("没有基类进行训练。跳过基类阶段。")
        base_model = None # 如果没有基类，则不创建模型
        accuracies_over_stages.append(0.0) # acc0 为 0

    # --- 增量学习阶段 ---
    current_known_orig_labels = list(base_class_labels) # 当前所有已知类的原始标签
    current_prototypes_remapped = {} # {全局重映射标签: 原型}

    if base_model: # 只有在基模型存在（即有基类）时才进行原型计算和增量学习
        # 首先为基类计算原型
        print("为基类计算初始原型 (使用整个基类训练验证数据)...")
        global_label_map_func = {orig_label: i for i, orig_label in enumerate(current_known_orig_labels)} # 初始全局映射
        
        base_data_all_torch = torch.tensor(base_data_np_train_val, dtype=torch.float32) # 使用之前划分的基类数据
        # 注意: base_labels_np_train_val_orig 是这些数据的原始标签
        
        # 分批提取基类特征
        base_features_list = []
        base_data_loader_for_proto = DataLoader(TensorDataset(base_data_all_torch), batch_size=BATCH_SIZE*2, shuffle=False)
        with torch.no_grad():
            for (batch_proto_data,) in tqdm(base_data_loader_for_proto, desc="提取基类原型特征", leave=False):
                base_features_list.append(base_model(batch_proto_data.to(DEVICE), extract_features=True))
        all_base_features = torch.cat(base_features_list, dim=0)

        for orig_label in base_class_labels:
            mask = (base_labels_np_train_val_orig == orig_label) # 使用原始标签进行掩码
            if mask.sum() > 0:
                prototype = all_base_features[mask].mean(dim=0)
                current_prototypes_remapped[global_label_map_func[orig_label]] = prototype
        print(f"基类原型计算完毕，数量: {len(current_prototypes_remapped)}")


    for stage_idx in range(actual_n_incremental_stages):
        print(f"\n--- 增量阶段 {stage_idx + 1} ---")
        new_stage_orig_labels = incremental_stages_orig_labels[stage_idx]
        
        # 更新已知类别列表和全局标签映射
        current_known_orig_labels.extend(new_stage_orig_labels)
        global_label_map_func = {orig_label: i for i, orig_label in enumerate(current_known_orig_labels)}
        num_total_known_classes_remapped = len(current_known_orig_labels)

        print(f"当前已知类别总数 (重映射后): {num_total_known_classes_remapped}")
        print(f"为阶段 {stage_idx + 1} 的新类别计算原型: {new_stage_orig_labels}")

        stage_data_indices = np.isin(all_labels_np, new_stage_orig_labels)
        stage_data_np = all_data_np[stage_data_indices]
        stage_labels_np_orig = all_labels_np[stage_data_indices] # 新类别的原始标签

        if len(stage_data_np) > 0 and base_model:
            stage_data_torch = torch.tensor(stage_data_np, dtype=torch.float32)
            
            # 分批提取新阶段类别特征
            stage_features_list = []
            stage_data_loader_for_proto = DataLoader(TensorDataset(stage_data_torch), batch_size=BATCH_SIZE*2, shuffle=False)
            with torch.no_grad():
                for (batch_proto_data,) in tqdm(stage_data_loader_for_proto, desc=f"提取增量阶段{stage_idx+1}原型特征", leave=False):
                    stage_features_list.append(base_model(batch_proto_data.to(DEVICE), extract_features=True))
            all_stage_features = torch.cat(stage_features_list, dim=0)


            for original_new_label in new_stage_orig_labels:
                mask = (stage_labels_np_orig == original_new_label)
                if mask.sum() > 0:
                    prototype = all_stage_features[mask].mean(dim=0)
                    current_prototypes_remapped[global_label_map_func[original_new_label]] = prototype
            print(f"当前原型总数: {len(current_prototypes_remapped)}")
        elif not base_model:
            print("基模型不存在，无法计算增量原型。")
            accuracies_over_stages.append(0.0) # 无法评估
            continue


        # --- 对当前所有已知类别进行测试 ---
        test_indices_current_stage = np.isin(all_labels_np, current_known_orig_labels)
        test_data_current_stage_np = all_data_np[test_indices_current_stage]
        test_labels_current_stage_np_orig = all_labels_np[test_indices_current_stage]

        if len(test_data_current_stage_np) == 0 or not base_model:
            print("当前阶段没有测试数据或基模型不存在，跳过评估。")
            accuracies_over_stages.append(accuracies_over_stages[-1] if accuracies_over_stages else 0.0)
            continue
        
        test_data_current_stage_tensor = torch.tensor(test_data_current_stage_np, dtype=torch.float32) # .to(DEVICE) in eval func
        test_labels_current_stage_remapped_tensor = torch.tensor(
            [global_label_map_func[orig_label] for orig_label in test_labels_current_stage_np_orig],
            dtype=torch.long
        ) # .to(DEVICE) in eval func

        print(f"在 {num_total_known_classes_remapped} 个已知类别上评估...")
        stage_accuracy = evaluate_with_prototypes(base_model,
                                                  test_data_current_stage_tensor,
                                                  test_labels_current_stage_remapped_tensor,
                                                  current_prototypes_remapped,
                                                  num_total_known_classes_remapped,
                                                  DEVICE)
        accuracies_over_stages.append(stage_accuracy)
        print(f"增量阶段 {stage_idx + 1} 准确率 (在所有 {num_total_known_classes_remapped} 个已知类别上): {stage_accuracy:.4f}")

    # 确保即使某些增量阶段没有进行，accuracies_over_stages列表也有预期的长度（用最后一个有效值或0填充）
    expected_len = 1 + actual_n_incremental_stages # acc0 + acc for each actual stage
    while len(accuracies_over_stages) < expected_len:
        accuracies_over_stages.append(accuracies_over_stages[-1] if accuracies_over_stages else 0.0)


    # --- 打印最终准确率 ---
    print("\n--- 最终准确率 ---")
    acc_str_parts = []
    if len(accuracies_over_stages) > 0:
        print(f"基类阶段准确率 (acc0): {accuracies_over_stages[0]:.4f}")
        acc_str_parts.append(f"{accuracies_over_stages[0]:.4f}")

    known_classes_count = len(base_class_labels)
    for i in range(actual_n_incremental_stages):
        if (i+1) < len(accuracies_over_stages): # 确保索引有效
            known_classes_count += len(incremental_stages_orig_labels[i])
            print(f"第 {i+1} 增量阶段后准确率 (acc{i+1}, {known_classes_count} 类): {accuracies_over_stages[i+1]:.4f}")
            acc_str_parts.append(f"{accuracies_over_stages[i+1]:.4f}")
        else: # 如果由于某种原因，某个阶段的准确率没有记录
            acc_str_parts.append("N/A")


    # 打印 acc0, acc1, acc2 格式
    # 如果阶段数少于2个增量阶段，则只打印可用的部分
    final_acc_print_str = ", ".join(acc_str_parts)
    print(f"\nacc0, acc1, ... : {final_acc_print_str}")

    print("\n--- 脚本执行完毕 ---")