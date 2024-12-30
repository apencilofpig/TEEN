from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging
from .MarginLoss import MarginLoss, margin_loss


def base_train(model, trainloader, optimizer, scheduler, epoch, args, fc):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    # 参数化设置
    num_splits = 4  # 划分为几份
    new_labels = [args.base_class + i for i in range(num_splits)]  # 新标签列表
    modifications = [
        {4: 0, 5: 0, 6: 72, 7: 68},  # 第1部分的修改规则
        {9: 1},  # 第2部分的修改规则
        {9: 1, 10: 1},
        {42: 114, 43: 113, 44: 185, 45: 67}
    ]
    # 检查参数一致性
    assert len(new_labels) == num_splits, "每一份需要指定一个新标签"
    assert len(modifications) == num_splits, "每一份需要指定修改规则"
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        # # 找出标签为 0 的样本索引
        # mask = (train_label == 0).squeeze()  # 标签为 0 的 mask, (batch_size,)

        # if mask.any():  # 检查是否有标签为 0 的样本
        #     # 1. 复制标签为0的样本
        #     new_data = data[mask].clone()  # 复制符合条件的样本
        #     new_train_label = train_label[mask].clone()  # 复制标签为0的样本标签

        #     # 2. 修改复制样本第3个维度（4, 5, 6, 7位置）上的值
        #     new_data[:, 0, 4] = 0   # 修改第4个位置的值为0
        #     new_data[:, 0, 5] = 0   # 修改第5个位置的值为0
        #     new_data[:, 0, 6] = 72  # 修改第6个位置的值为72
        #     new_data[:, 0, 7] = 68  # 修改第7个位置的值为68
        #     new_data[:, 0, 9] = 1

        #     # 3. 修改新样本的标签
        #     new_train_label[:] = 26  # 将标签修改为新的值

        #     # 4. 将新样本和原始 batch 合并
        #     data = torch.cat([data, new_data], dim=0)  # 合并数据
        #     train_label = torch.cat([train_label, new_train_label], dim=0)  # 合并标签

        # 找出标签为 0 的样本索引
        mask = (train_label == 0).squeeze()  # 标签为 0 的 mask, (batch_size,)

        if mask.any():  # 检查是否有标签为 0 的样本
            # 1. 复制标签为 0 的样本
            zero_class_data = data[mask].clone()  # 提取标签为0的样本
            zero_class_label = train_label[mask].clone()  # 提取对应标签

            # 2. 随机划分为指定份数
            num_samples = zero_class_data.size(0)
            indices = torch.randperm(num_samples)  # 随机打乱样本索引
            splits = torch.chunk(indices, num_splits)  # 按指定份数划分

            # 3. 遍历每一份，修改样本和标签
            new_data_list, new_labels_list = [], []
            for split_idx, split_indices in enumerate(splits):
                # 提取当前划分的样本
                current_data = zero_class_data[split_indices]
                current_label = zero_class_label[split_indices]

                # 修改样本值
                for dim, value in modifications[split_idx].items():
                    current_data[:, 0, dim] = value

                # 修改标签
                current_label[:] = new_labels[split_idx]

                # 保存修改后的样本和标签
                new_data_list.append(current_data)
                new_labels_list.append(current_label)

            # 4. 合并所有新生成的样本和标签
            new_data = torch.cat(new_data_list, dim=0)
            new_labels = torch.cat(new_labels_list, dim=0)
            data = torch.cat([data, new_data], dim=0)  # 合并到原始数据中
            train_label = torch.cat([train_label, new_labels], dim=0)  # 合并到原始标签中



        logits = model(data)
        logits = logits[:, :args.base_class+num_splits]

        # proto_0 = fc.weight[:, 0]
        # dists = F.cosine_similarity(fc.weight[:, 1:], proto_0.unsqueeze(1), dim=0)

        # loss = F.cross_entropy(logits, train_label) + dists.mean()
        loss = F.cross_entropy(logits, train_label)
        # loss = margin_loss(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

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
            model.module.mode = 'encoder'
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

    model.module.fc.weight.data[:args.base_class] = proto_list

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

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(num_classes):
                class_mask = (labels == i)
                correct_per_class[i] += (preds[class_mask] == labels[class_mask]).sum().item()
                total_per_class[i] += class_mask.sum().item()

    acc_per_class = [correct / total if total != 0 else 0 for correct, total in zip(correct_per_class, total_per_class)]
    return acc_per_class

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
            outputs = model(images)
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


def test(model, testloader, epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    print(get_accuracy_per_class(model, testloader, test_class))
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5= Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt=torch.cat([lgt,logits.cpu()])
            lbs=torch.cat([lbs,test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5= va5.item()
        
        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        lgt=lgt.view(-1, test_class)
        lbs=lbs.view(-1)

        # if session > 0:
        #     _preds = torch.argmax(lgt, dim=1)
        #     torch.save(_preds, f"pred_labels/{args.project}_{args.dataset}_{session}_preds.pt")
        #     torch.save(lbs, f"pred_labels/{args.project}_{args.dataset}_{session}_labels.pt")
        #     torch.save(model.module.fc.weight.data.cpu()[:test_class], f"pred_labels/{args.project}_{args.dataset}_{session}_weights.pt")
            
        if session > 0:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt,lbs)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            
            result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac}")
            return vl, (seenac, unseenac, va)
        else:
            return vl, va
