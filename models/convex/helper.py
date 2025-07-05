from dataloader.swat import swat
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging

# +++ START OF ADDED IMPORTS FOR CVXPY +++
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import cvxpy as cp
# +++ END OF ADDED IMPORTS FOR CVXPY +++


# +++ START OF ADDED CVXPY AND DATA FUNCTIONS +++
def get_swat_data_for_cvxpy(args):
    """Loads and preprocesses the entire SWaT dataset for CVXPY initialization."""
    logging.info("Loading full SWaT dataset for CVXPY initialization...")
    try:
        df = pd.read_csv(args.csv_path, header=None)
        data = swat.base_inputs_train
        labels = swat.base_labels_train
    except Exception as e:
        raise FileNotFoundError(f"Could not load SWaT CSV file from {args.csv_path}. Error: {e}")

    # Scale data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Load attack info to calculate alpha_sensor
    try:
        with open(args.json_path, 'r') as f:
            attack_info = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load attack JSON file from {args.json_path}. Error: {e}")

    alpha_counts = np.zeros(args.num_features_input)
    for item in attack_info:
        for point_idx in item['points']:
            if 0 <= point_idx < args.num_features_input:
                alpha_counts[point_idx] += 1
    
    if np.sum(alpha_counts) == 0:
        logging.warning("Alpha counts are all zero. Using uniform alpha.")
        alpha_counts = np.ones(args.num_features_input)
    
    alpha_sensor_np = alpha_counts / np.sum(alpha_counts)
    logging.info(f"Alpha (sensor importance sum): {alpha_sensor_np.sum()}")

    return data, labels, alpha_sensor_np

def calculate_W_init_cvxpy(data_np, labels_np, alpha_sensor_np, args):
    """Calculates the initial weight matrix W_init using CVXPY."""
    logging.info("Starting CVXPY optimization to calculate W_init...")
    
    num_features = args.num_features_input
    W_init_cvx = cp.Variable((num_features, num_features), name="W_init")
    alpha_cvx = alpha_sensor_np

    # Subsample data to make the problem tractable
    selected_indices = []
    unique_labels = np.unique(labels_np)
    for label in unique_labels:
        class_indices = np.where(labels_np == label)[0]
        if len(class_indices) > 0:
            selected_indices.extend(np.random.choice(
                class_indices,
                size=min(len(class_indices), args.cvxpy_samples_per_class),
                replace=False
            ))

    if not selected_indices:
        logging.warning("No data selected for CVXPY. Returning random W_init.")
        return np.random.randn(num_features, num_features).astype(np.float32)

    cvx_data = data_np[selected_indices]
    cvx_labels = labels_np[selected_indices]
    num_cvx_samples = len(cvx_labels)
    
    W0_np = np.eye(num_features, dtype=np.float32)
    
    term1_intra_class_diff = []
    term2_inter_class_grad_sum = np.zeros_like(W0_np)
    pair_count_intra, pair_count_inter = 0, 0

    # Intra-class term
    for i in tqdm(range(num_cvx_samples), desc="CVXPY Intra-class Pairs", leave=False):
        for j in range(i + 1, num_cvx_samples):
            if cvx_labels[i] == cvx_labels[j]:
                if pair_count_intra < args.max_cvx_pairs:
                    Vij = (cvx_data[i] - cvx_data[j]).reshape(-1, 1)
                    term1_intra_class_diff.append(alpha_cvx @ cp.abs(W_init_cvx @ Vij))
                    pair_count_intra += 1
                else: break
        if pair_count_intra >= args.max_cvx_pairs: break

    # Inter-class term
    for i in tqdm(range(num_cvx_samples), desc="CVXPY Inter-class Pairs", leave=False):
        for j in range(num_cvx_samples):
            if cvx_labels[i] != cvx_labels[j]:
                if pair_count_inter < args.max_cvx_pairs:
                    Vij = (cvx_data[i] - cvx_data[j]).reshape(-1, 1)
                    W0_Vij = W0_np @ Vij
                    grad_f1_contrib = (alpha_cvx * np.sign(W0_Vij.flatten()))[:, np.newaxis] @ Vij.T
                    term2_inter_class_grad_sum += grad_f1_contrib
                    pair_count_inter += 1
                else: break
        if pair_count_inter >= args.max_cvx_pairs: break

    if not term1_intra_class_diff:
        logging.warning("No intra-class pairs for CVXPY objective. Returning random W_init.")
        return np.random.randn(num_features, num_features).astype(np.float32)

    objective_expr = cp.sum(term1_intra_class_diff)
    if pair_count_inter > 0:
        objective_expr -= args.cvxpy_lambda * cp.trace(term2_inter_class_grad_sum.T @ W_init_cvx)

    constraints = [cp.norm(W_init_cvx, "nuc") <= args.cvxpy_c1_limit]
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    
    logging.info("Solving CVXPY problem...")
    try:
        problem.solve(solver=cp.SCS, verbose=False, max_iters=2500)
        if W_init_cvx.value is not None:
            logging.info(f"CVXPY solve completed. Status: {problem.status}")
            return W_init_cvx.value.astype(np.float32)
        else:
            raise cp.error.SolverError("Solver finished but solution is null.")
    except Exception as e:
        logging.error(f"CVXPY failed: {e}. Returning random W_init.")
        return np.random.randn(num_features, num_features).astype(np.float32)
# +++ END OF ADDED CVXPY AND DATA FUNCTIONS +++


def base_train(model, trainloader, optimizer, scheduler, epoch, args, fc):
    # This function remains unchanged from your original framework
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        # Use the model's forward pass
        x_feature, logits = model(data)
        logits = logits[:, :args.base_class]
        
        total_loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

# ... (the rest of your helper.py file remains unchanged)
def test(model, testloader, epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    logging.info(get_accuracy_per_class(model, testloader, test_class))
    model = model.eval()
    vl = Averager()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    va5= Averager()
    vacc = Averager()  # 准确率平均
    vprecision = Averager()  # 精确率平均
    vrecall = Averager()  # 召回率平均
    vf1 = Averager()  # F1平均
    lgt=torch.tensor([])  # logits
    lbs=torch.tensor([])  # labels
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            x_feature, logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs]) # 只预测基类的情况下基类预测正确的数量
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs]) # 给出新类的情况下基类预测正确的数量
                va_base.add(acc_base) # 不断计算平均准确率
                va_base_given_new.add(acc_base_given_new)


            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class) # 只预测新类的情况下新类预测正确的数量
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs]) # 给出基类的情况下基类预测正确的数量
                va_new.add(acc_new)
                va_new_given_base.add(acc_new_given_base)

            labels = test_label.cpu().numpy()
            _, preds = torch.max(logits, 1)
            preds = preds.cpu().numpy()

            # logging.info(labels)
            # logging.info(preds)
            vacc.add(accuracy_score(labels, preds))
            vprecision.add(precision_score(labels, preds, average='macro', zero_division=0))
            vrecall.add(recall_score(labels, preds, average='macro', zero_division=0))
            vf1.add(f1_score(labels, preds, average='macro', zero_division=0))

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5= va5.item()
        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
        vacc = vacc.item()
        vprecision = vprecision.item()
        vrecall = vrecall.item()
        vf1 = vf1.item()
        
        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}, accuracy={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'.format(epoch, vl, va, va5, vacc, vprecision, vrecall, vf1))

        lgt=lgt.view(-1, test_class)
        lbs=lbs.view(-1)

        if session > 0:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt,lbs)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            
            result_list.append(f"Seen Acc:{va_base_given_new}  Unseen Acc:{va_new_given_base}")
            return vl, (va_base_given_new, va_new_given_base, va, vacc, vprecision, vrecall, vf1)
        else:
            return vl, va, vacc, vprecision, vrecall, vf1

# ... other functions like replace_base_fc, get_accuracy_per_class, etc. remain here

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class] = proto_list

    return model

import torch

def get_accuracy_per_class(model, testloader, num_classes):
    model.eval()
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            _, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(num_classes):
                class_mask = (labels == i)
                correct_per_class[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                total_per_class[i] += class_mask.sum().item()

    acc_per_class = [correct / total if total != 0 else 0 for correct, total in zip(correct_per_class, total_per_class)]
    return acc_per_class

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def get_metrics(model, testloader, num_classes):
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            _, logits = model(inputs)
            logits = logits[:, :num_classes]
            _, preds = torch.max(logits, 1)

        logging.info(count_acc(preds, labels))
        
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        # 计算准确率（多分类任务中直接使用）
        accuracy = accuracy_score(labels, preds)

        # 计算精确率（多分类任务中需要指定 average 参数）
        precision = precision_score(labels, preds, average='macro')  # 宏平均

        # 计算召回率（多分类任务中需要指定 average 参数）
        recall = recall_score(labels, preds, average='macro')  

        # 计算 F1 分数（多分类任务中需要指定 average 参数）
        f1 = f1_score(labels, preds, average='macro')  

        logging.info(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_accuracy_confusion_matrix(model, testloader, num_classes, save_path):
    """
    计算模型在测试数据集上的准确率和混淆矩阵,并可视化混淆矩阵。
    
    参数:
    model (nn.Module): 要评估的深度学习模型
    testloader (DataLoader): 测试数据集的数据加载器
    num_classes (int): 分类问题的类别数量
    
    返回:
    accuracy (float): 模型在测试数据集上的准确率
    cm (numpy.ndarray): 模型的混淆矩阵
    """

    # 创建一个函数来格式化单元格值
    def format_cell(value):
        return f"{value:.1f}"

    # 设置模型为评估模式
    model.eval()
    
    # 初始化预测标签和真实标签列表
    y_true = []
    y_pred = []
    
    # 在测试数据集上进行预测
    with torch.no_grad():
        for images, labels in testloader:
            _, outputs = model(images.to('cuda'))
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    
    # 计算准确率
    accuracy = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100).astype('int') # 归一化
    
    # 更改默认字体
    plt.rcParams['font.family'] = 'Ubuntu'  # 这里以 'SimHei' 字体为例
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(8, 8), dpi=800)
    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
    disp.plot(cmap='Blues', ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks(np.arange(len(disp.display_labels)))
    ax.set_yticks(np.arange(len(disp.display_labels)))
    ax.set_xticklabels(disp.display_labels)
    ax.set_yticklabels(disp.display_labels)
    ax.tick_params(axis='both', which='major', pad=10)
    plt.savefig(save_path)
    
    return accuracy, cm